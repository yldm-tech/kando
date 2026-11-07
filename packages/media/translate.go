package media

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"kando/packages/core"
)

// translateBatch bounds how many lines go into one request, keeping replies within common output token limits.
const translateBatch = 40

// DefaultSubtitlePrompt is the style instruction used when ChatTranslator.Prompt is empty. {{language}} becomes the target language. The JSON reply rule is appended separately and is not part of this text.
const DefaultSubtitlePrompt = "You are a professional subtitle translator. Translate every subtitle line into {{language}}.\n- Keep each translation natural, concise and suitable for on-screen subtitles.\n- Lines may be sentence fragments that continue in the next line; translate them so they read naturally in order."

const subtitleReplyRule = "The user sends a JSON array of subtitle lines in order. Reply with only a JSON array of strings with exactly the same number of elements, each the translation of the line at the same index. Do not merge, split, or skip lines."

// contextLines is how many earlier lines of the same video go with a batch when context is on. Subtitle text is short, but an entire film can still crowd a small model.
const contextLines = 80

// ChatTranslator translates through an OpenAI-compatible /chat/completions endpoint, such as EveryAPI, OpenAI or a local Ollama server.
type ChatTranslator struct {
	BaseURL string
	APIKey  string
	Model   string
	// Prompt is the style instruction. {{language}} is replaced with the target language. Empty uses DefaultSubtitlePrompt.
	Prompt string
	// Context includes earlier lines of the same video with each batch.
	Context bool
	// Credential, when set, replaces BaseURL and APIKey for each translation. renew asks for a new key after the API rejected the previous one.
	Credential func(ctx context.Context, renew bool) (Endpoint, error)
	// Client defaults to an http.Client with a two-minute timeout.
	Client *http.Client
}

// Endpoint is where translation requests go and the bearer key they carry.
type Endpoint struct {
	BaseURL string
	APIKey  string
}

// Ping checks that the chat endpoint accepts one short request with the configured model.
func (t ChatTranslator) Ping(ctx context.Context) error {
	if strings.TrimSpace(t.Model) == "" {
		return errors.New("model is empty")
	}
	endpoint := Endpoint{BaseURL: strings.TrimSpace(t.BaseURL), APIKey: t.APIKey}
	if t.Credential != nil {
		var err error
		if endpoint, err = t.Credential(ctx, false); err != nil {
			return err
		}
	}
	if strings.TrimSpace(endpoint.BaseURL) == "" {
		return errors.New("base URL is empty")
	}
	body, err := json.Marshal(chatRequest{
		Model:       t.Model,
		Temperature: 0,
		Messages:    []chatMessage{{Role: "user", Content: "ping"}},
	})
	if err != nil {
		return err
	}
	client := t.Client
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	_, _, err = completeOnce(ctx, client, endpoint, body)
	return err
}

func (t ChatTranslator) Translate(ctx context.Context, segments []core.Segment, language string) ([]core.Segment, error) {
	if strings.TrimSpace(language) == "" {
		return nil, fmt.Errorf("target language is empty")
	}
	endpoint := Endpoint{BaseURL: t.BaseURL, APIKey: t.APIKey}
	if t.Credential != nil {
		var err error
		if endpoint, err = t.Credential(ctx, false); err != nil {
			return nil, err
		}
	}
	renewed := false
	out := make([]core.Segment, 0, len(segments))
	var earlier []string
	// untranslated counts the lines of batches left untranslated, failedRun the batches in a row that failed, and lastErr why the last of them did.
	var untranslated, failedRun int
	var lastErr error
	batches := (len(segments) + translateBatch - 1) / translateBatch
	for start := 0; start < len(segments); start += translateBatch {
		batch := segments[start:min(start+translateBatch, len(segments))]
		progress := core.Fraction(float64(start) / float64(len(segments)))
		progress.Batch, progress.Batches, progress.BatchSize = start/translateBatch+1, batches, translateBatch
		reportProgress(ctx, progress)
		texts := make([]string, len(batch))
		for i, s := range batch {
			texts[i] = s.Text
		}
		var contextSoFar []string
		if t.Context {
			contextSoFar = earlier
		}
		lines, err := t.translateTexts(ctx, endpoint, texts, language, contextSoFar, progress)
		earlier = append(earlier, texts...)
		var status *statusError
		if t.Credential != nil && !renewed && errors.As(err, &status) && status.code == http.StatusUnauthorized {
			renewed = true
			if endpoint, err = t.Credential(ctx, true); err != nil {
				return nil, err
			}
			lines, err = t.translateTexts(ctx, endpoint, texts, language, contextSoFar, progress)
		}
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			// A rejected key, model or address fails every batch the same way, as do batches that fail one after another; one batch that fails after its retries is left untranslated and the rest of the film goes on.
			failedRun++
			if !skippableBatch(err) || failedRun >= maxFailedBatches {
				return nil, err
			}
			lastErr = err
			untranslated += len(batch)
			lines = make([]string, len(batch))
		} else {
			failedRun = 0
		}
		for i, s := range batch {
			s.Text = lines[i]
			out = append(out, s)
		}
	}
	if untranslated == len(segments) && lastErr != nil {
		return nil, lastErr
	}
	reportProgress(ctx, core.Fraction(1))
	if untranslated > 0 {
		return out, &UntranslatedError{Lines: untranslated, Total: len(segments), Err: lastErr}
	}
	return out, nil
}

// maxFailedBatches is how many batches in a row may fail before the translation stops: that many are the service's trouble, not their lines'.
const maxFailedBatches = 3

// UntranslatedError is what Translate returns, with every line, when it left some lines untranslated (an empty Text) because their batch still failed after it was tried again. Lines of Total are untranslated; Err is why the last of their batches failed.
type UntranslatedError struct {
	Lines, Total int
	Err          error
}

func (e *UntranslatedError) Error() string {
	return fmt.Sprintf("%d of %d lines were left untranslated: %v", e.Lines, e.Total, e.Err)
}

func (e *UntranslatedError) Unwrap() error { return e.Err }

// skippableBatch reports whether a batch that failed with err can be left untranslated while the rest of the film goes on: a reply that could not be read or had the wrong lines, a request that timed out or lost its connection, a rate limit or a server error that outlasted its retries. A key, model or address the API rejects is not: it fails every batch.
func skippableBatch(err error) bool {
	var status *statusError
	if errors.As(err, &status) {
		return status.code == http.StatusTooManyRequests || status.code == http.StatusRequestTimeout || status.code >= 500
	}
	return true
}

// Chat sends one conversation, a system instruction and the user's message, to the same endpoint and model translations use, and returns the model's reply without the code fence a model may wrap around it. A key the API rejects is renewed once through Credential, as Translate does; rate limits, server errors, timeouts and lost connections are retried as for a translation.
func (t ChatTranslator) Chat(ctx context.Context, system, user string) (string, error) {
	if strings.TrimSpace(t.Model) == "" {
		return "", errors.New("model is empty")
	}
	endpoint := Endpoint{BaseURL: strings.TrimSpace(t.BaseURL), APIKey: t.APIKey}
	if t.Credential != nil {
		var err error
		if endpoint, err = t.Credential(ctx, false); err != nil {
			return "", err
		}
	}
	if strings.TrimSpace(endpoint.BaseURL) == "" {
		return "", errors.New("base URL is empty")
	}
	body, err := json.Marshal(chatRequest{
		Model:    t.Model,
		Messages: []chatMessage{{Role: "system", Content: system}, {Role: "user", Content: user}},
	})
	if err != nil {
		return "", err
	}
	reply, err := t.complete(ctx, endpoint, body, core.Progress{})
	var status *statusError
	if t.Credential != nil && errors.As(err, &status) && status.code == http.StatusUnauthorized {
		if endpoint, err = t.Credential(ctx, true); err != nil {
			return "", err
		}
		reply, err = t.complete(ctx, endpoint, body, core.Progress{})
	}
	if err != nil {
		return "", err
	}
	if len(reply.Choices) == 0 {
		return "", errors.New("chat API returned no choices")
	}
	return stripCodeFence(reply.Choices[0].Message.Content), nil
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

// statusError is a non-200 reply from the translation API.
type statusError struct {
	code   int
	status string
	detail string
}

// StatusCode is the HTTP status the API answered with.
func (e *statusError) StatusCode() int { return e.code }

func (e *statusError) Error() string {
	return fmt.Sprintf("translation API returned %s: %s", e.status, e.detail)
}

func subtitlePrompt(custom, language string, earlier []string) string {
	prompt := strings.TrimSpace(custom)
	if prompt == "" {
		prompt = DefaultSubtitlePrompt
	}
	prompt = strings.ReplaceAll(prompt, "{{language}}", language)
	if len(earlier) > contextLines {
		earlier = earlier[len(earlier)-contextLines:]
	}
	if len(earlier) > 0 {
		encoded, _ := json.Marshal(earlier)
		prompt += "\n\nEarlier lines from the same video, for context only. Do not translate them again:\n" + string(encoded)
	}
	return prompt + "\n\n" + subtitleReplyRule
}

// maxBatchSends is how many times one batch is sent while its reply is not the batch's lines: not a JSON array of them, or too few or too many.
const maxBatchSends = 3

// translateTexts sends one batch, again while the reply is not its lines, up to maxBatchSends times. progress is where the translation stands while it is sent, which a retry's wait is reported with.
func (t ChatTranslator) translateTexts(ctx context.Context, endpoint Endpoint, texts []string, language string, earlier []string, progress core.Progress) ([]string, error) {
	input, err := json.Marshal(texts)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(chatRequest{
		Model: t.Model,
		Messages: []chatMessage{
			{Role: "system", Content: subtitlePrompt(t.Prompt, language, earlier)},
			{Role: "user", Content: string(input)},
		},
	})
	if err != nil {
		return nil, err
	}

	for send := 1; ; send++ {
		reply, err := t.complete(ctx, endpoint, body, progress)
		if err != nil {
			return nil, err
		}
		lines, err := replyLines(reply, len(texts))
		if err == nil {
			return lines, nil
		}
		if send == maxBatchSends || ctx.Err() != nil {
			// Models occasionally omit only a few lines from a large response.
			// Retry those responses as smaller batches so one malformed reply does
			// not discard an otherwise usable translation.
			var countErr *lineCountError
			if errors.As(err, &countErr) && recoverableLineCount(countErr.got, countErr.want) {
				mid := len(texts) / 2
				left, leftErr := t.translateTexts(ctx, endpoint, texts[:mid], language, earlier, progress)
				if leftErr != nil {
					return nil, leftErr
				}
				rightEarlier := append(append([]string(nil), earlier...), texts[:mid]...)
				right, rightErr := t.translateTexts(ctx, endpoint, texts[mid:], language, rightEarlier, progress)
				if rightErr != nil {
					return nil, rightErr
				}
				return append(left, right...), nil
			}
			return lines, err
		}
	}
}

// recoverableLineCount identifies a nearly complete response worth retrying as
// smaller requests. A one-line reply for a two-line batch is usually an
// instruction failure, and splitting it would duplicate an unknown translation.
func recoverableLineCount(got, want int) bool {
	return want > 2 && got > 1 && got*2 >= want
}

// replyLines reads the n translated lines of a batch from the model's reply: a JSON array of n strings, maybe in a code fence. An element that is an array of strings, as a model writes for a half sentence split across two lines, is that line's translation in parts, which are joined.
func replyLines(reply chatResponse, n int) ([]string, error) {
	if len(reply.Choices) == 0 {
		return nil, fmt.Errorf("translation API returned no choices")
	}
	var raw []json.RawMessage
	if err := json.Unmarshal([]byte(stripCodeFence(reply.Choices[0].Message.Content)), &raw); err != nil {
		return nil, fmt.Errorf("translation is not a JSON array of strings: %w", err)
	}
	if len(raw) != n {
		return nil, &lineCountError{got: len(raw), want: n}
	}
	lines := make([]string, n)
	for i, element := range raw {
		if err := json.Unmarshal(element, &lines[i]); err == nil {
			continue
		}
		var parts []string
		if err := json.Unmarshal(element, &parts); err != nil {
			return nil, fmt.Errorf("translation is not a JSON array of strings: line %d is %s", i+1, element)
		}
		lines[i] = joinParts(parts)
	}
	return lines, nil
}

type lineCountError struct {
	got, want int
}

func (e *lineCountError) Error() string {
	return fmt.Sprintf("translation returned %d lines for %d", e.got, e.want)
}

// joinParts joins the parts of one translated line: with nothing between Chinese or Japanese text, which writes no spaces, and with a space between words otherwise.
func joinParts(parts []string) string {
	var b strings.Builder
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if b.Len() > 0 {
			last, _ := utf8.DecodeLastRuneInString(b.String())
			first, _ := utf8.DecodeRuneInString(part)
			if !unspaced(last) && !unspaced(first) {
				b.WriteByte(' ')
			}
		}
		b.WriteString(part)
	}
	return b.String()
}

// unspaced reports whether r belongs to a script written without spaces between words, or is the full-width punctuation that goes with it.
func unspaced(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Thai) || (r >= 0x3000 && r <= 0x303F) || (r >= 0xFF00 && r <= 0xFFEF)
}

// TranslateFail is why a translation failed, read from its error by ClassifyTranslate, so the card can say it in the user's words. The empty kind is a failure it does not recognize.
type TranslateFail string

const (
	// TranslateAuth is a key the API rejected (401 or 403), even after it was renewed.
	TranslateAuth TranslateFail = "auth"
	// TranslateRejected is a request the API refused for another reason of the client's, as a model or an address it does not know.
	TranslateRejected TranslateFail = "rejected"
	// TranslateBusy is a rate limit or a server error that outlasted the retries.
	TranslateBusy TranslateFail = "busy"
	// TranslateTimeout is a request that got no answer in time, after the retries.
	TranslateTimeout TranslateFail = "timeout"
	// TranslateNetwork is a request that could not reach the API or lost its connection, after the retries.
	TranslateNetwork TranslateFail = "network"
	// TranslateMalformed is a reply that was not the batch's lines, after the batch was sent again.
	TranslateMalformed TranslateFail = "malformed"
)

// statusPattern finds the HTTP status in a statusError's text.
var statusPattern = regexp.MustCompile(`translation API returned (\d{3}) `)

// translateSigns are the phrases of the errors this file and net/http give, in the order they are tried: a timeout while a reply's body was read is a timeout before it is a reply that could not be decoded, and a connection cut off while it was read is the network's.
var translateSigns = []struct {
	kind    TranslateFail
	phrases []string
}{
	{TranslateTimeout, []string{"Client.Timeout", "deadline exceeded", "i/o timeout", "TLS handshake timeout", "timeout awaiting response headers"}},
	{TranslateNetwork, []string{"no such host", "connection refused", "connection reset", "network is unreachable", "no route to host", "broken pipe", "server misbehaving", "unexpected EOF", ": EOF"}},
	{TranslateMalformed, []string{"not a JSON array of strings", "lines for", "returned no choices", "decode translation response"}},
}

// ClassifyTranslate reads why a translation failed from its error's text, which is what a failed job keeps once it has been saved and read back. It returns "" for an error it does not recognize.
func ClassifyTranslate(msg string) TranslateFail {
	if m := statusPattern.FindStringSubmatch(msg); m != nil {
		code, _ := strconv.Atoi(m[1])
		switch {
		case code == http.StatusUnauthorized || code == http.StatusForbidden:
			return TranslateAuth
		case code == http.StatusTooManyRequests || code >= 500:
			return TranslateBusy
		case code == http.StatusRequestTimeout:
			return TranslateTimeout
		}
		return TranslateRejected
	}
	for _, sign := range translateSigns {
		for _, phrase := range sign.phrases {
			if strings.Contains(msg, phrase) {
				return sign.kind
			}
		}
	}
	return ""
}

const (
	maxAttempts   = 3
	maxRetryAfter = 30 * time.Second
)

// retryBaseDelay doubles after each failed attempt unless the server sends Retry-After.
var retryBaseDelay = 2 * time.Second

// complete posts body to the chat endpoint, retrying rate limits and server errors, which gateways return when upstream capacity is briefly exhausted, and requests that timed out, lost their connection or came back cut off, while ctx lasts. Before each retry it reports progress with how long it waits, as the server's Retry-After asked or the backoff chose, and which retry comes; once the wait is over it reports progress as it was.
func (t ChatTranslator) complete(ctx context.Context, endpoint Endpoint, body []byte, progress core.Progress) (chatResponse, error) {
	client := t.Client
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Minute}
	}
	delay := retryBaseDelay
	for attempt := 1; ; attempt++ {
		reply, retryAfter, err := completeOnce(ctx, client, endpoint, body)
		if err == nil || retryAfter < 0 || attempt == maxAttempts {
			return reply, err
		}
		var status *statusError
		wait := delay
		if retryAfter > 0 {
			wait = min(retryAfter, maxRetryAfter)
		}
		delay *= 2
		waiting := progress
		waiting.RetryIn, waiting.Attempt, waiting.Attempts = wait, attempt, maxAttempts-1
		waiting.Unanswered = !errors.As(err, &status)
		reportProgress(ctx, waiting)
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return chatResponse{}, ctx.Err()
		case <-timer.C:
		}
		reportProgress(ctx, progress)
	}
}

// completeOnce makes one request. A non-negative retryAfter marks a retryable failure, positive when the server said how long to wait: a rate limit, a server error, or a request that got no usable answer, as one that timed out, lost its connection or came back cut off. Once ctx is done its error is returned as it is, not retried.
func completeOnce(ctx context.Context, client *http.Client, endpoint Endpoint, body []byte) (reply chatResponse, retryAfter time.Duration, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(endpoint.BaseURL, "/")+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return reply, -1, err
	}
	req.Header.Set("Content-Type", "application/json")
	if endpoint.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+endpoint.APIKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return reply, -1, ctxErr
		}
		return reply, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		err := &statusError{code: resp.StatusCode, status: resp.Status, detail: strings.TrimSpace(string(detail))}
		if resp.StatusCode != http.StatusTooManyRequests && resp.StatusCode < 500 {
			return reply, -1, err
		}
		retryAfter = 0
		if secs, convErr := strconv.Atoi(resp.Header.Get("Retry-After")); convErr == nil && secs > 0 {
			retryAfter = time.Duration(secs) * time.Second
		}
		return reply, retryAfter, err
	}
	if err := json.NewDecoder(resp.Body).Decode(&reply); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return reply, -1, ctxErr
		}
		return reply, 0, fmt.Errorf("decode translation response: %w", err)
	}
	return reply, -1, nil
}

// stripCodeFence removes the ```json fence that chat models often wrap around JSON.
func stripCodeFence(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	s = strings.TrimPrefix(s, "```")
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[i+1:]
	}
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), "```"))
}
