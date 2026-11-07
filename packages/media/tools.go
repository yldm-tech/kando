package media

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"kando/packages/core"
)

// Tools runs the pipeline with yt-dlp, ffmpeg and whisper.cpp installed on this machine, and translates with an OpenAI-compatible chat API.
type Tools struct {
	// OutputDir receives downloaded videos and their subtitles.
	OutputDir string
	// DownloadTemp holds what yt-dlp writes while a video downloads (the separate video and sound tracks of a split format, .part files), one folder per source, so the library under OutputDir only ever sees the merged video. Empty uses a folder in the system's temp folder. When it is not on the library's volume a hidden folder of the library is used instead (downloadTempFor).
	DownloadTemp string
	// YTDLP, FFmpeg and Whisper are executable paths; empty means not found.
	YTDLP   string
	FFmpeg  string
	Whisper string
	// WhisperModel is the path of a ggml model file for whisper.cpp.
	WhisperModel string
	Translator   ChatTranslator
	// Cookies is the browser whose login Download borrows, as yt-dlp's --cookies-from-browser names it (chrome, safari, firefox, edge, brave); empty borrows none.
	Cookies string
	// OEmbed is YouTube's oEmbed endpoint, which names a video yt-dlp could not look up, as one that is age-restricted or blocked where this machine is; empty asks nothing.
	OEmbed string
}

// EveryAPI is the default translation provider; setting KANDO_LLM_BASE_URL switches to any other OpenAI-compatible provider.
const (
	EveryAPIBaseURL = "https://api.everyapi.ai/v1"
	EveryAPIModel   = "claude-haiku-5-5"
)

// ToolsFromEnv configures Tools from KANDO_* environment variables and the executables it can find.
func ToolsFromEnv() Tools {
	outputDir := os.Getenv("KANDO_OUTPUT_DIR")
	if outputDir == "" {
		outputDir = DefaultLibraryDir()
	}
	translator := ChatTranslator{
		BaseURL: os.Getenv("KANDO_LLM_BASE_URL"),
		APIKey:  os.Getenv("KANDO_LLM_API_KEY"),
		Model:   os.Getenv("KANDO_LLM_MODEL"),
	}
	// EveryAPI defaults apply only without a custom provider, so its key and model names never reach another API.
	if translator.BaseURL == "" {
		translator.BaseURL = EveryAPIBaseURL
		if translator.APIKey == "" {
			translator.APIKey = os.Getenv("EVERYAPI_API_KEY")
		}
		if translator.APIKey == "" {
			if cli := lookTool("everyapi"); cli != "" {
				translator.Credential = everyAPICredential(cli)
			}
		}
		if translator.Model == "" {
			translator.Model = EveryAPIModel
		}
	}
	downloadTemp := ""
	if cache, err := os.UserCacheDir(); err == nil {
		downloadTemp = filepath.Join(cache, "Kando", "downloads")
	}
	return Tools{
		OutputDir:    outputDir,
		DownloadTemp: downloadTemp,
		YTDLP:        lookTool("yt-dlp"),
		FFmpeg:       lookTool("ffmpeg"),
		Whisper:      lookTool("whisper-cli"),
		WhisperModel: os.Getenv("KANDO_WHISPER_MODEL"),
		Translator:   translator,
		OEmbed:       YouTubeOEmbed,
	}
}

// LibraryExt is Kando's own library bundle suffix.
const LibraryExt = ".kando"

// DefaultLibraryDir is the bundle that holds one folder per video.
func DefaultLibraryDir() string {
	return filepath.Join(moviesDir(), "影视库"+LibraryExt)
}

// VimpoLibraryDir is the other app's library. Kando does not store its videos there.
func VimpoLibraryDir() string {
	return filepath.Join(moviesDir(), "影视库.vimpo")
}

// MoviesDir is the folder a new library is created in unless the user picks another place.
func MoviesDir() string { return moviesDir() }

func moviesDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return os.TempDir()
	}
	folder := "Videos"
	if runtime.GOOS == "darwin" {
		folder = "Movies"
	}
	return filepath.Join(home, folder)
}

// Apps opened from Finder or a desktop launcher get a minimal PATH that omits package manager directories.
var extraToolDirs = []string{"/opt/homebrew/bin", "/usr/local/bin", "/home/linuxbrew/.linuxbrew/bin"}

func lookTool(name string) string {
	if path, err := exec.LookPath(name); err == nil {
		return path
	}
	for _, dir := range extraToolDirs {
		path := filepath.Join(dir, name)
		if info, err := os.Stat(path); err == nil && !info.IsDir() && info.Mode()&0o111 != 0 {
			return path
		}
	}
	return ""
}

// Relook finds yt-dlp, ffmpeg and whisper-cli again. The output directory and the translator stay as they are.
func (t Tools) Relook() Tools {
	t.YTDLP = lookTool("yt-dlp")
	t.FFmpeg = lookTool("ffmpeg")
	t.Whisper = lookTool("whisper-cli")
	return t
}

func (t Tools) Check() error {
	var errs []error
	for _, tool := range []struct{ name, path string }{{"yt-dlp", t.YTDLP}, {"ffmpeg", t.FFmpeg}, {"whisper-cli", t.Whisper}} {
		if tool.path == "" {
			errs = append(errs, fmt.Errorf("%s is not installed", tool.name))
		}
	}
	if t.WhisperModel == "" {
		errs = append(errs, errors.New("KANDO_WHISPER_MODEL is not set"))
	} else if _, err := os.Stat(t.WhisperModel); err != nil {
		errs = append(errs, fmt.Errorf("whisper model: %w", err))
	}
	if t.Translator.Model == "" {
		errs = append(errs, errors.New("KANDO_LLM_MODEL is not set"))
	}
	if t.Translator.Credential != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		_, err := t.Translator.Credential(ctx, false)
		cancel()
		if err != nil {
			errs = append(errs, err)
		}
	} else if t.Translator.BaseURL == EveryAPIBaseURL && t.Translator.APIKey == "" {
		errs = append(errs, errors.New("EveryAPI is not signed in; run `everyapi auth login` or set EVERYAPI_API_KEY"))
	}
	return errors.Join(errs...)
}

// Download fetches source with yt-dlp into its own folder under OutputDir and returns the video's path. While it runs it reports to the context's Trace the video's title and thumbnail, once yt-dlp has looked the video up, and how far the download has gone. When yt-dlp fails before it could say what the video is called, a YouTube video's title and thumbnail are asked of OEmbed, so the card that says why it failed can name it.
//
// yt-dlp downloads into a folder kept for source (downloadTempFor), in DownloadTemp or a hidden folder of the library, and moves only the merged video into the library, where the library watcher would otherwise take a finished video track for a video of its own while the sound track is still downloading (GatherVideos). The folder is removed once the video is in the library and kept when the download fails, so a retry resumes it.
func (t Tools) Download(ctx context.Context, source string) (string, error) {
	if path, ok := localVideoPath(source); ok {
		if _, err := os.Stat(path); err != nil {
			return "", err
		}
		return path, nil
	}
	temp, err := t.downloadTempFor(source)
	if err != nil {
		return "", err
	}
	named := false
	path, err := t.ytdlpDownload(ctx, source, temp, &named)
	// An earlier Kando renamed the video track of a download that had not merged yet to video.<ext> and left the other track beside it; yt-dlp takes that silent file for the finished video. Leftovers beside the video mean it never merged, so it is removed with them and downloaded again.
	if err == nil && holdsDownloadLeftovers(filepath.Dir(path)) && pathInside(t.OutputDir, path) {
		if err = removeUnmerged(path); err == nil {
			path, err = t.ytdlpDownload(ctx, source, temp, &named)
		}
	}
	if err != nil {
		if !named && ctx.Err() == nil && t.OEmbed != "" {
			if info, ok := lookupOEmbed(ctx, t.OEmbed, source); ok {
				reportInfo(ctx, info)
			}
		}
		return "", err
	}
	_ = os.RemoveAll(temp)
	// The library's hidden folder goes too once no other download is left in it; a folder that still holds one is not empty and stays.
	if filepath.Base(filepath.Dir(temp)) == libraryDownloads {
		_ = os.Remove(filepath.Dir(temp))
	}
	return path, nil
}

// ytdlpDownload runs yt-dlp once for source, with temp as the folder of its partial files, and returns the path it says the video was saved to. named is set once yt-dlp has said what the video is called.
func (t Tools) ytdlpDownload(ctx context.Context, source, temp string, named *bool) (string, error) {
	reader := &ytdlpReader{
		progress: func(p core.Progress) { reportProgress(ctx, p) },
		info: func(info Info) {
			*named = *named || info.Title != ""
			reportInfo(ctx, info)
		},
	}
	args := []string{
		"--no-playlist",
		"-f", "bv*+ba/b",
		"--merge-output-format", "mp4",
		"-P", t.OutputDir,
		// A relative temp path would be taken as inside the library.
		"-P", "temp:" + temp,
		"-o", "%(title).80B [%(id)s]/video.%(ext)s",
		"--print", ytdlpInfoTemplate,
		"--print", ytdlpFormatsTemplate,
		"--print", "after_move:filepath",
		// --print implies --quiet, which hides the progress unless asked for again.
		"--progress", "--newline",
		"--progress-template", ytdlpProgressTemplate,
	}
	// yt-dlp looks for ffmpeg, which merges the video with its sound, on PATH, which an app opened from Finder does not extend to Homebrew's folder; without it the video is saved silent.
	if t.FFmpeg != "" {
		args = append(args, "--ffmpeg-location", t.FFmpeg)
	}
	if t.Cookies != "" {
		args = append(args, "--cookies-from-browser", t.Cookies)
	}
	if _, _, err := runFollowing(ctx, false, reader.line, t.YTDLP, append(args, "--", source)...); err != nil {
		return "", err
	}
	if reader.path == "" {
		return "", errors.New("yt-dlp did not report the downloaded file")
	}
	return reader.path, nil
}

// libraryDownloads is the hidden folder of the library that holds partial downloads when DownloadTemp is not on the library's volume. The library's readers skip names that start with a dot.
const libraryDownloads = ".kando-downloads"

// downloadTempFor is the absolute folder that holds source's partial download, the same for the same source and library. It is a folder of DownloadTemp when that is on the library's volume, and otherwise one in the library's hidden libraryDownloads: yt-dlp moves the merged video into the library with a rename only on the same volume, and across volumes copies it into video.<ext> in place, so a download stopped during the copy would leave a cut-off video that the next try takes for a finished one.
func (t Tools) downloadTempFor(source string) (string, error) {
	sum := sha256.Sum256([]byte(source))
	name := hex.EncodeToString(sum[:8])
	base := t.DownloadTemp
	if base == "" {
		base = filepath.Join(os.TempDir(), "kando-downloads")
	}
	base, err := filepath.Abs(base)
	if err != nil {
		return "", err
	}
	library, err := filepath.Abs(t.OutputDir)
	if err != nil {
		return "", err
	}
	if os.MkdirAll(base, 0o755) == nil && sameVolume(base, library) {
		return filepath.Join(base, name), nil
	}
	return filepath.Join(library, libraryDownloads, name), nil
}

// sameVolume reports whether directories a and b are known to be on one volume. A platform without directory identities (IdentifyDir) never knows.
func sameVolume(a, b string) bool {
	devA, _, okA := IdentifyDir(a).Parts()
	devB, _, okB := IdentifyDir(b).Parts()
	return okA && okB && devA == devB
}

// removeUnmerged removes video and the files beside it that a download leaves before it merges (downloadLeftover).
func removeUnmerged(video string) error {
	folder := filepath.Dir(video)
	entries, err := os.ReadDir(folder)
	if err != nil {
		return err
	}
	remove := []string{video}
	for _, entry := range entries {
		if !entry.IsDir() && downloadLeftover(entry.Name()) {
			remove = append(remove, filepath.Join(folder, entry.Name()))
		}
	}
	for _, path := range remove {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

// localVideoPath reports the disk path of a file:// video the user picked. Other URLs are downloaded.
func localVideoPath(source string) (string, bool) {
	parsed, err := url.Parse(source)
	if err != nil || parsed.Scheme != "file" || parsed.Path == "" || (parsed.Host != "" && parsed.Host != "localhost") {
		return "", false
	}
	return parsed.Path, true
}

func (t Tools) Transcribe(ctx context.Context, videoPath string) ([]core.Segment, error) {
	if strings.TrimSpace(videoPath) == "" {
		return nil, errors.New("video path is empty")
	}
	dir, err := os.MkdirTemp("", "kando-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)

	// whisper.cpp expects 16 kHz mono 16-bit PCM.
	audio := filepath.Join(dir, "audio.wav")
	if _, err := run(ctx, t.FFmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", videoPath, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", audio); err != nil {
		return nil, err
	}
	transcript := filepath.Join(dir, "transcript")
	// -pp prints how far it has gone on stderr, which -np leaves alone.
	follow := func(line string) {
		if f, ok := whisperFraction(line); ok {
			reportProgress(ctx, core.Fraction(f))
		}
	}
	if _, _, err := runFollowing(ctx, true, follow, t.Whisper, "-m", t.WhisperModel, "-t", strconv.Itoa(whisperThreadCount()), "-l", "auto", "-np", "-pp", "-oj", "-of", transcript, "-f", audio); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(transcript + ".json")
	if err != nil {
		return nil, err
	}
	return parseWhisperJSON(data)
}

// whisperThreadCount uses the available CPU without letting a large machine
// create more decoder contention than whisper.cpp can use effectively. Four is
// whisper-cli's default, but it leaves most modern Macs idle during decoding.
func whisperThreadCount() int {
	n := runtime.NumCPU()
	if n < 1 {
		return 1
	}
	if n > 8 {
		return 8
	}
	return n
}

func (t Tools) Translate(ctx context.Context, segments []core.Segment, language string) ([]core.Segment, error) {
	return t.Translator.Translate(ctx, segments, language)
}

func (t Tools) Export(ctx context.Context, videoPath string, original, translated []core.Segment) (string, error) {
	return exportSubtitles(ctx, videoPath, original, translated)
}

// everyAPICredentialVersion is the output version of `everyapi auth credential` this package understands; the CLI bumps it only for incompatible changes.
const everyAPICredentialVersion = 1

// everyAPICredential reads the signed-in EveryAPI CLI's relay key through `everyapi auth credential`, which refreshes OAuth keys under its own lock and resolves the gateway region. renew invalidates a key the gateway rejected.
func everyAPICredential(cli string) func(context.Context, bool) (Endpoint, error) {
	return func(ctx context.Context, renew bool) (Endpoint, error) {
		args := []string{"auth", "credential", "--format=json"}
		if renew {
			args = append(args, "--invalidate")
		}
		out, err := run(ctx, cli, args...)
		if err != nil {
			if strings.Contains(err.Error(), "EVERYAPI_CREDENTIAL_ERROR:not_logged_in") {
				return Endpoint{}, errors.New("EveryAPI is not signed in; run `everyapi auth login`")
			}
			return Endpoint{}, fmt.Errorf("EveryAPI credential: %w", err)
		}
		var cred struct {
			Version int    `json:"version"`
			BaseURL string `json:"base_url"`
			APIKey  string `json:"api_key"`
		}
		if err := json.Unmarshal([]byte(out), &cred); err != nil {
			return Endpoint{}, fmt.Errorf("EveryAPI credential: %w", err)
		}
		if cred.Version != everyAPICredentialVersion {
			return Endpoint{}, fmt.Errorf("EveryAPI credential version %d is not supported; update Kando", cred.Version)
		}
		if cred.BaseURL == "" || cred.APIKey == "" {
			return Endpoint{}, errors.New("EveryAPI credential is missing its base URL or key")
		}
		return Endpoint{BaseURL: cred.BaseURL, APIKey: cred.APIKey}, nil
	}
}

// run executes a tool and returns its stdout, folding the end of stderr into the error.
func run(ctx context.Context, name string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", ctxErr
		}
		return "", fmt.Errorf("%s: %w: %s", filepath.Base(name), err, toolError(stderr.String()))
	}
	return stdout.String(), nil
}

func lastLine(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	return strings.TrimSpace(lines[len(lines)-1])
}

func tail(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return "..." + strings.ToValidUTF8(s[len(s)-n:], "")
}
