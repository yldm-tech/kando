package media

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"kando/packages/core"
)

// fakeChat answers each request with reply(lines), recording the requests it saw.
func fakeChat(t *testing.T, reply func(lines []string) string) (*httptest.Server, *[]chatRequest) {
	t.Helper()
	var seen []chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer secret" {
			http.Error(w, "unexpected request "+r.URL.Path, http.StatusUnauthorized)
			return
		}
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		seen = append(seen, req)
		var lines []string
		if err := json.Unmarshal([]byte(req.Messages[1].Content), &lines); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": reply(lines)}}},
		})
	}))
	t.Cleanup(srv.Close)
	return srv, &seen
}

func upper(lines []string) string {
	out := make([]string, len(lines))
	for i, l := range lines {
		out[i] = strings.ToUpper(l)
	}
	data, _ := json.Marshal(out)
	return "```json\n" + string(data) + "\n```"
}

func TestChatTranslatorBatchesAndKeepsTiming(t *testing.T) {
	srv, seen := fakeChat(t, upper)
	segments := make([]core.Segment, translateBatch+3)
	for i := range segments {
		segments[i] = core.Segment{Start: time.Duration(i) * time.Second, End: time.Duration(i+1) * time.Second, Text: fmt.Sprintf("line %d", i)}
	}

	tr := ChatTranslator{BaseURL: srv.URL + "/v1/", APIKey: "secret", Model: "test-model"}
	got, err := tr.Translate(context.Background(), segments, "English")
	if err != nil {
		t.Fatal(err)
	}
	if len(*seen) != 2 {
		t.Fatalf("made %d requests, want 2", len(*seen))
	}
	if req := (*seen)[0]; req.Model != "test-model" || !strings.Contains(req.Messages[0].Content, "English") {
		t.Fatalf("unexpected request: %#v", req)
	}
	if len(got) != len(segments) {
		t.Fatalf("got %d segments", len(got))
	}
	for i, s := range got {
		if s.Text != fmt.Sprintf("LINE %d", i) || s.Start != segments[i].Start || s.End != segments[i].End {
			t.Fatalf("segment %d = %#v", i, s)
		}
	}
}

func TestChatTranslatorRejectsWrongLineCount(t *testing.T) {
	srv, _ := fakeChat(t, func([]string) string { return `["only one"]` })
	tr := ChatTranslator{BaseURL: srv.URL + "/v1", APIKey: "secret", Model: "m"}
	_, err := tr.Translate(context.Background(), []core.Segment{{Text: "a"}, {Text: "b"}}, "English")
	if err == nil || !strings.Contains(err.Error(), "1 lines for 2") {
		t.Fatalf("err = %v", err)
	}
}

func TestChatTranslatorReportsHTTPErrors(t *testing.T) {
	srv, _ := fakeChat(t, upper)
	tr := ChatTranslator{BaseURL: srv.URL + "/v1", APIKey: "wrong", Model: "m"}
	_, err := tr.Translate(context.Background(), []core.Segment{{Text: "a"}}, "English")
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("err = %v", err)
	}
}

// flakyChat fails with the given statuses in order, then translates by echoing.
func flakyChat(t *testing.T, statuses ...int) (*httptest.Server, *int) {
	t.Helper()
	old := retryBaseDelay
	retryBaseDelay = time.Millisecond
	t.Cleanup(func() { retryBaseDelay = old })
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls <= len(statuses) {
			http.Error(w, `{"error":{"code":"channel:capacity_limited"}}`, statuses[calls-1])
			return
		}
		var req chatRequest
		json.NewDecoder(r.Body).Decode(&req)
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": req.Messages[1].Content}}},
		})
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func TestChatTranslatorRetriesCapacityErrors(t *testing.T) {
	srv, calls := flakyChat(t, http.StatusTooManyRequests, http.StatusServiceUnavailable)
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	got, err := tr.Translate(context.Background(), []core.Segment{{Text: "hello"}}, "English")
	if err != nil {
		t.Fatal(err)
	}
	if *calls != 3 || got[0].Text != "hello" {
		t.Fatalf("calls = %d, got %#v", *calls, got)
	}
}

func TestChatTranslatorGivesUpAfterMaxAttempts(t *testing.T) {
	srv, calls := flakyChat(t, http.StatusTooManyRequests, http.StatusTooManyRequests, http.StatusTooManyRequests, http.StatusTooManyRequests)
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	_, err := tr.Translate(context.Background(), []core.Segment{{Text: "hello"}}, "English")
	if err == nil || !strings.Contains(err.Error(), "capacity_limited") || *calls != maxAttempts {
		t.Fatalf("calls = %d, err = %v", *calls, err)
	}
}

func TestChatTranslatorDoesNotRetryClientErrors(t *testing.T) {
	srv, calls := flakyChat(t, http.StatusUnauthorized)
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	if _, err := tr.Translate(context.Background(), []core.Segment{{Text: "hello"}}, "English"); err == nil || *calls != 1 {
		t.Fatalf("calls = %d, err = %v", *calls, err)
	}
}

func TestChatTranslatorStopsRetryingWhenCancelled(t *testing.T) {
	srv, _ := flakyChat(t, http.StatusTooManyRequests, http.StatusTooManyRequests)
	retryBaseDelay = time.Hour
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	if _, err := tr.Translate(ctx, []core.Segment{{Text: "hello"}}, "English"); err != context.DeadlineExceeded {
		t.Fatalf("err = %v", err)
	}
}

func TestChatTranslatorRenewsRejectedLogin(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Header.Get("Authorization"))
		if len(seen) == 1 {
			http.Error(w, `{"error":{"code":"invalid_api_key"}}`, http.StatusUnauthorized)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": `["hello"]`}}},
		})
	}))
	t.Cleanup(srv.Close)
	var renews int
	tr := ChatTranslator{
		Model: "m",
		Credential: func(_ context.Context, renew bool) (Endpoint, error) {
			renews++
			key := "old"
			if renew {
				key = "new"
			}
			return Endpoint{BaseURL: srv.URL, APIKey: key}, nil
		},
	}
	got, err := tr.Translate(context.Background(), []core.Segment{{Text: "a"}}, "English")
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Text != "hello" || renews != 2 {
		t.Fatalf("renews = %d, got %#v", renews, got)
	}
	if !reflect.DeepEqual(seen, []string{"Bearer old", "Bearer new"}) {
		t.Fatalf("authorization = %v", seen)
	}
}

func TestTranslatePromptAndContext(t *testing.T) {
	srv, seen := fakeChat(t, upper)
	segments := make([]core.Segment, translateBatch+1)
	for i := range segments {
		segments[i] = core.Segment{Text: fmt.Sprintf("line %d", i)}
	}
	tr := ChatTranslator{
		BaseURL: srv.URL + "/v1", APIKey: "secret", Model: "m", Context: true,
		Prompt: "Translate into {{language}} in a formal tone.",
	}
	if _, err := tr.Translate(context.Background(), segments, "日本語"); err != nil {
		t.Fatal(err)
	}
	first := (*seen)[0].Messages[0].Content
	second := (*seen)[1].Messages[0].Content
	if !strings.Contains(first, "Translate into 日本語 in a formal tone.") || strings.Contains(first, "Earlier lines") {
		t.Fatalf("first prompt = %q", first)
	}
	if !strings.Contains(second, "line 0") || !strings.Contains(second, "Earlier lines") {
		t.Fatalf("second prompt = %q", second)
	}
}

func TestChatTranslatorRequiresLanguage(t *testing.T) {
	if _, err := (ChatTranslator{}).Translate(context.Background(), []core.Segment{{Text: "a"}}, " "); err == nil {
		t.Fatal("expected an error")
	}
}

func TestPing(t *testing.T) {
	if err := (ChatTranslator{BaseURL: "https://example.com/v1"}).Ping(context.Background()); err == nil {
		t.Fatal("expected an empty model to fail")
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer secret" {
			http.Error(w, "no", http.StatusUnauthorized)
			return
		}
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Model != "m" || len(req.Messages) != 1 {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": "ok"}}},
		})
	}))
	defer srv.Close()
	tr := ChatTranslator{BaseURL: srv.URL + "/v1", APIKey: "secret", Model: "m", Client: srv.Client()}
	if err := tr.Ping(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestChatTranslatorReportsEachBatchAndRetryWait(t *testing.T) {
	srv, _ := flakyChat(t, http.StatusTooManyRequests)
	segments := make([]core.Segment, translateBatch+3)
	for i := range segments {
		segments[i] = core.Segment{Text: fmt.Sprintf("line %d", i)}
	}
	var steps []core.Progress
	ctx := WithTrace(context.Background(), &Trace{Progress: func(p core.Progress) { steps = append(steps, p) }})
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	if _, err := tr.Translate(ctx, segments, "English"); err != nil {
		t.Fatal(err)
	}
	// Batch 1 of 2 starts, waits once for the rate limit, carries on, then batch 2 starts and the translation ends.
	if len(steps) != 5 {
		t.Fatalf("steps = %+v", steps)
	}
	first, waiting, resumed, second, done := steps[0], steps[1], steps[2], steps[3], steps[4]
	if first.Batch != 1 || first.Batches != 2 || first.BatchSize != translateBatch || first.Fraction != 0 || first.RetryIn != 0 {
		t.Fatalf("first = %+v", first)
	}
	if waiting.RetryIn <= 0 || waiting.Attempt != 1 || waiting.Attempts != maxAttempts-1 || waiting.Batch != 1 {
		t.Fatalf("waiting = %+v", waiting)
	}
	if resumed.RetryIn != 0 || resumed.Batch != 1 {
		t.Fatalf("resumed = %+v", resumed)
	}
	if second.Batch != 2 || second.Fraction != float64(translateBatch)/float64(len(segments)) {
		t.Fatalf("second = %+v", second)
	}
	if !done.Known || done.Fraction != 1 {
		t.Fatalf("done = %+v", done)
	}
}

// TestChatSendsTheConversationAndReturnsTheReply checks that Chat posts the system and user messages to the configured endpoint with its key and model, and returns the model's reply as it wrote it, without the code fence a model may wrap around it.
func TestChatSendsTheConversationAndReturnsTheReply(t *testing.T) {
	var seen chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer secret" {
			http.Error(w, "unexpected request", http.StatusUnauthorized)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&seen); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": "```json\n{\"answer\":\"ok\"}\n```"}}},
		})
	}))
	t.Cleanup(srv.Close)
	tr := ChatTranslator{BaseURL: srv.URL + "/v1", APIKey: "secret", Model: "m1"}
	got, err := tr.Chat(context.Background(), "be brief", "why?")
	if err != nil {
		t.Fatal(err)
	}
	if got != `{"answer":"ok"}` {
		t.Fatalf("reply %q", got)
	}
	want := []chatMessage{{Role: "system", Content: "be brief"}, {Role: "user", Content: "why?"}}
	if seen.Model != "m1" || !reflect.DeepEqual(seen.Messages, want) {
		t.Fatalf("request %+v", seen)
	}
}

// TestChatRenewsRejectedLogin checks that Chat asks the credential for a new key once after the API rejected the first, as Translate does.
func TestChatRenewsRejectedLogin(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer fresh" {
			http.Error(w, "expired", http.StatusUnauthorized)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": "hi"}}},
		})
	}))
	t.Cleanup(srv.Close)
	var renewed []bool
	tr := ChatTranslator{Model: "m1", Credential: func(_ context.Context, renew bool) (Endpoint, error) {
		renewed = append(renewed, renew)
		key := "stale"
		if renew {
			key = "fresh"
		}
		return Endpoint{BaseURL: srv.URL, APIKey: key}, nil
	}}
	got, err := tr.Chat(context.Background(), "s", "u")
	if err != nil || got != "hi" {
		t.Fatalf("reply %q err %v", got, err)
	}
	if !reflect.DeepEqual(renewed, []bool{false, true}) {
		t.Fatalf("credential asked %v", renewed)
	}
}

// TestChatNeedsAModelAndAnEndpoint checks that Chat says what is missing rather than posting nowhere.
func TestChatNeedsAModelAndAnEndpoint(t *testing.T) {
	if _, err := (ChatTranslator{BaseURL: "http://127.0.0.1:1"}).Chat(context.Background(), "s", "u"); err == nil {
		t.Fatal("no error without a model")
	}
	if _, err := (ChatTranslator{Model: "m"}).Chat(context.Background(), "s", "u"); err == nil {
		t.Fatal("no error without a base URL")
	}
}

// replyChat answers the n-th request (from 0) with the chat reply content replies[n], the last one again after them, recording how many requests it saw.
func replyChat(t *testing.T, replies ...string) (*httptest.Server, *int) {
	t.Helper()
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		content := replies[min(calls, len(replies)-1)]
		calls++
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": content}}},
		})
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

// A request the API does not answer in time is sent again, as a rate limit is, and the batch it carried is translated.
func TestChatTranslatorRetriesATimeout(t *testing.T) {
	old := retryBaseDelay
	retryBaseDelay = time.Millisecond
	t.Cleanup(func() { retryBaseDelay = old })
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			time.Sleep(300 * time.Millisecond)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": `["你好"]`}}},
		})
	}))
	t.Cleanup(srv.Close)
	var waits []core.Progress
	ctx := WithTrace(context.Background(), &Trace{Progress: func(p core.Progress) {
		if p.RetryIn > 0 {
			waits = append(waits, p)
		}
	}})
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m", Client: &http.Client{Timeout: 100 * time.Millisecond}}
	got, err := tr.Translate(ctx, []core.Segment{{Text: "hello"}}, "简体中文")
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || got[0].Text != "你好" {
		t.Fatalf("calls = %d, got %#v", calls, got)
	}
	if len(waits) != 1 || !waits[0].Unanswered {
		t.Fatalf("waits = %+v, want one wait for a request that was not answered", waits)
	}
}

// A reply cut off before its JSON ends is asked for again rather than failing the film.
func TestChatTranslatorRetriesAReplyItCannotRead(t *testing.T) {
	old := retryBaseDelay
	retryBaseDelay = time.Millisecond
	t.Cleanup(func() { retryBaseDelay = old })
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"[\"你`))
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": `["你好"]`}}},
		})
	}))
	t.Cleanup(srv.Close)
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	got, err := tr.Translate(context.Background(), []core.Segment{{Text: "hello"}}, "简体中文")
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || got[0].Text != "你好" {
		t.Fatalf("calls = %d, got %#v", calls, got)
	}
}

// A model sometimes answers a half sentence split across two cues with a nested array for one of them; its parts are that line's translation, joined.
func TestChatTranslatorJoinsANestedLine(t *testing.T) {
	srv, calls := replyChat(t, `["我们去", ["河边", "看看"], "好吗"]`)
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	got, err := tr.Translate(context.Background(), []core.Segment{{Text: "let's go"}, {Text: "to the river and"}, {Text: "see, ok?"}}, "简体中文")
	if err != nil {
		t.Fatal(err)
	}
	if *calls != 1 || got[1].Text != "河边看看" {
		t.Fatalf("calls = %d, got %#v", *calls, got)
	}
	if joined := joinParts([]string{"to the river", "and look"}); joined != "to the river and look" {
		t.Fatalf("joined words %q", joined)
	}
}

// A reply that is not the batch's lines, too few of them or not an array at all, is asked for again.
func TestChatTranslatorResendsAMalformedBatch(t *testing.T) {
	srv, calls := replyChat(t, `["只有一句"]`, `好的，以下是翻译`, `["一", "二"]`)
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	got, err := tr.Translate(context.Background(), []core.Segment{{Text: "one"}, {Text: "two"}}, "简体中文")
	if err != nil {
		t.Fatal(err)
	}
	if *calls != 3 || got[0].Text != "一" || got[1].Text != "二" {
		t.Fatalf("calls = %d, got %#v", *calls, got)
	}
}

// A nearly complete large reply is split into smaller requests after retries,
// so a single omitted line does not fail the whole film.
func TestChatTranslatorSplitsANearlyCompleteBatch(t *testing.T) {
	srv, seen := fakeChat(t, func(lines []string) string {
		if len(lines) == 8 {
			return upper(lines[:7])
		}
		return upper(lines)
	})
	segments := make([]core.Segment, 8)
	for i := range segments {
		segments[i] = core.Segment{Text: fmt.Sprintf("line %d", i)}
	}
	tr := ChatTranslator{BaseURL: srv.URL + "/v1", APIKey: "secret", Model: "m"}
	got, err := tr.Translate(context.Background(), segments, "English")
	if err != nil {
		t.Fatal(err)
	}
	if len(*seen) != maxBatchSends+2 || len(got) != len(segments) {
		t.Fatalf("requests = %d, segments = %d", len(*seen), len(got))
	}
	for i, s := range got {
		if s.Text != fmt.Sprintf("LINE %d", i) {
			t.Fatalf("segment %d = %#v", i, s)
		}
	}
}

// A batch that still fails after it was sent again stays untranslated, the rest of the film is translated, and the error says how many lines were left.
func TestChatTranslatorLeavesAFailedBatchUntranslated(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var req chatRequest
		json.NewDecoder(r.Body).Decode(&req)
		var lines []string
		json.Unmarshal([]byte(req.Messages[1].Content), &lines)
		content := upper(lines)
		if lines[0] == "line 0" {
			content = `["broken"]`
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": content}}},
		})
	}))
	t.Cleanup(srv.Close)
	segments := make([]core.Segment, translateBatch+3)
	for i := range segments {
		segments[i] = core.Segment{Start: time.Duration(i) * time.Second, Text: fmt.Sprintf("line %d", i)}
	}
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	got, err := tr.Translate(context.Background(), segments, "English")
	var partial *UntranslatedError
	if !errors.As(err, &partial) || partial.Lines != translateBatch || partial.Total != len(segments) {
		t.Fatalf("err = %v", err)
	}
	if calls != maxBatchSends+1 || len(got) != len(segments) {
		t.Fatalf("calls = %d, %d segments", calls, len(got))
	}
	if got[0].Text != "" || got[translateBatch-1].Text != "" || got[translateBatch].Text != fmt.Sprintf("LINE %d", translateBatch) || got[translateBatch].Start != segments[translateBatch].Start {
		t.Fatalf("got %#v", got)
	}
}

// Batches that fail one after another are the service's trouble, not their lines': the translation stops rather than sending every batch of the film in vain.
func TestChatTranslatorStopsAfterFailedBatchesInARow(t *testing.T) {
	srv, calls := replyChat(t, `not json`)
	segments := make([]core.Segment, translateBatch*(maxFailedBatches+2))
	for i := range segments {
		segments[i] = core.Segment{Text: "a"}
	}
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	_, err := tr.Translate(context.Background(), segments, "English")
	var partial *UntranslatedError
	if err == nil || errors.As(err, &partial) || ClassifyTranslate(err.Error()) != TranslateMalformed {
		t.Fatalf("err = %v", err)
	}
	if *calls != maxFailedBatches*maxBatchSends {
		t.Fatalf("calls = %d", *calls)
	}
}

// A key the API rejects is not a batch's trouble: the translation fails at once.
func TestChatTranslatorFailsAtOnceOnARejectedKey(t *testing.T) {
	srv, calls := flakyChat(t, http.StatusForbidden)
	tr := ChatTranslator{BaseURL: srv.URL, Model: "m"}
	segments := make([]core.Segment, translateBatch*2)
	_, err := tr.Translate(context.Background(), segments, "English")
	if err == nil || *calls != 1 || ClassifyTranslate(err.Error()) != TranslateAuth {
		t.Fatalf("calls = %d, err = %v", *calls, err)
	}
}

func TestClassifyTranslate(t *testing.T) {
	for msg, want := range map[string]TranslateFail{
		"translation API returned 401 Unauthorized: invalid key":                                                                         TranslateAuth,
		"translation API returned 403 Forbidden: ":                                                                                       TranslateAuth,
		"translation API returned 404 Not Found: no such model":                                                                          TranslateRejected,
		"translation API returned 400 Bad Request: model is required":                                                                    TranslateRejected,
		"translation API returned 429 Too Many Requests: slow down":                                                                      TranslateBusy,
		"translation API returned 503 Service Unavailable: capacity_limited":                                                             TranslateBusy,
		`Post "https://api.example.com/v1/chat/completions": context deadline exceeded (Client.Timeout exceeded while awaiting headers)`: TranslateTimeout,
		"decode translation response: context deadline exceeded (Client.Timeout or context cancellation while reading body)":             TranslateTimeout,
		`Post "https://api.example.com/v1/chat/completions": dial tcp: lookup api.example.com: no such host`:                             TranslateNetwork,
		`Post "http://127.0.0.1:1/chat/completions": dial tcp 127.0.0.1:1: connect: connection refused`:                                  TranslateNetwork,
		"decode translation response: unexpected EOF":                                                                                    TranslateNetwork,
		"decode translation response: invalid character 'c' looking for beginning of value":                                              TranslateMalformed,
		"translation is not a JSON array of strings: invalid character '好'":                                                              TranslateMalformed,
		"translation returned 3 lines for 40":                                                                                            TranslateMalformed,
		"translation API returned no choices":                                                                                            TranslateMalformed,
		"something nobody has seen":                                                                                                      "",
	} {
		if got := ClassifyTranslate("translate: " + msg); got != want {
			t.Errorf("%q: %q, want %q", msg, got, want)
		}
	}
}

// TestChatTranslatorListsModels checks that Models asks the endpoint's /models with the key, returns the ids sorted, and renews a key the API rejects once through Credential, as Chat does.
func TestChatTranslatorListsModels(t *testing.T) {
	var auth []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		auth = append(auth, r.Header.Get("Authorization"))
		if r.Header.Get("Authorization") != "Bearer fresh" {
			http.Error(w, `{"error":"expired"}`, http.StatusUnauthorized)
			return
		}
		_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"zeta-chat"},{"id":"alpha-chat"},{"id":""},{"id":"alpha-chat"}]}`)
	}))
	defer server.Close()
	renewed := false
	translator := ChatTranslator{Credential: func(_ context.Context, renew bool) (Endpoint, error) {
		key := "stale"
		if renew {
			renewed, key = true, "fresh"
		}
		return Endpoint{BaseURL: server.URL + "/v1/", APIKey: key}, nil
	}}
	models, err := translator.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !renewed || !slices.Equal(models, []string{"alpha-chat", "zeta-chat"}) {
		t.Fatalf("renewed %v models %q auth %q", renewed, models, auth)
	}
	if _, err := (ChatTranslator{BaseURL: server.URL + "/v1", APIKey: "wrong"}).Models(context.Background()); err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("a rejected key: %v", err)
	}
	if _, err := (ChatTranslator{}).Models(context.Background()); err == nil {
		t.Fatal("no base URL gave no error")
	}
}
