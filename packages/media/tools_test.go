package media

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"kando/packages/core"
)

func TestToolsCheckListsEveryProblem(t *testing.T) {
	err := Tools{}.Check()
	if err == nil {
		t.Fatal("expected an error")
	}
	for _, want := range []string{"yt-dlp", "ffmpeg", "whisper-cli", "KANDO_WHISPER_MODEL", "KANDO_LLM_MODEL"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %s", err, want)
		}
	}
}

func TestToolsCheckPassesWhenConfigured(t *testing.T) {
	model := filepath.Join(t.TempDir(), "ggml-small.bin")
	if err := os.WriteFile(model, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	tools := Tools{YTDLP: "/bin/yt-dlp", FFmpeg: "/bin/ffmpeg", Whisper: "/bin/whisper-cli", WhisperModel: model, Translator: ChatTranslator{Model: "m"}}
	if err := tools.Check(); err != nil {
		t.Fatal(err)
	}
}

func setLLMEnv(t *testing.T, baseURL, apiKey, model, everyAPIKey string) {
	t.Setenv("KANDO_LLM_BASE_URL", baseURL)
	t.Setenv("KANDO_LLM_API_KEY", apiKey)
	t.Setenv("KANDO_LLM_MODEL", model)
	t.Setenv("EVERYAPI_API_KEY", everyAPIKey)
}

func TestToolsFromEnv(t *testing.T) {
	t.Setenv("KANDO_OUTPUT_DIR", "/out")
	t.Setenv("KANDO_WHISPER_MODEL", "/models/small.bin")
	setLLMEnv(t, "", "", "", "")
	tools := ToolsFromEnv()
	if tools.OutputDir != "/out" || tools.WhisperModel != "/models/small.bin" {
		t.Fatalf("unexpected tools: %#v", tools)
	}
}

func TestToolsFromEnvTranslator(t *testing.T) {
	for _, tc := range []struct {
		name                                string
		baseURL, apiKey, model, everyAPIKey string
		want                                ChatTranslator
	}{
		{
			name:        "EveryAPI by default",
			everyAPIKey: "sk-everyapi",
			want:        ChatTranslator{BaseURL: EveryAPIBaseURL, APIKey: "sk-everyapi", Model: EveryAPIModel},
		},
		{
			name:        "EveryAPI with chosen key and model",
			apiKey:      "sk-kando",
			model:       "doubao-seed-2.0-mini",
			everyAPIKey: "sk-everyapi",
			want:        ChatTranslator{BaseURL: EveryAPIBaseURL, APIKey: "sk-kando", Model: "doubao-seed-2.0-mini"},
		},
		{
			name:        "custom provider ignores EveryAPI defaults",
			baseURL:     "http://localhost:11434/v1",
			everyAPIKey: "sk-everyapi",
			want:        ChatTranslator{BaseURL: "http://localhost:11434/v1"},
		},
		{
			name:    "custom provider",
			baseURL: "https://llm.example.com/v1",
			apiKey:  "key",
			model:   "m",
			want:    ChatTranslator{BaseURL: "https://llm.example.com/v1", APIKey: "key", Model: "m"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setLLMEnv(t, tc.baseURL, tc.apiKey, tc.model, tc.everyAPIKey)
			got := ToolsFromEnv().Translator
			if got.Credential != nil || got.BaseURL != tc.want.BaseURL || got.APIKey != tc.want.APIKey || got.Model != tc.want.Model {
				t.Fatalf("translator = %#v, want %#v", got, tc.want)
			}
		})
	}
}

func TestToolsCheckRequiresEveryAPIKey(t *testing.T) {
	tools := Tools{Translator: ChatTranslator{BaseURL: EveryAPIBaseURL, Model: EveryAPIModel}}
	if err := tools.Check(); err == nil || !strings.Contains(err.Error(), "everyapi auth login") {
		t.Fatalf("err = %v", err)
	}
	tools.Translator = ChatTranslator{BaseURL: "http://localhost:11434/v1", Model: "qwen3"}
	if err := tools.Check(); err != nil && strings.Contains(err.Error(), "everyapi auth login") {
		t.Fatalf("custom provider asked for an EveryAPI login: %v", err)
	}
}

func writeFakeEveryAPI(t *testing.T, script string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "everyapi")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
}

func TestToolsFromEnvUsesSignedInCLI(t *testing.T) {
	writeFakeEveryAPI(t, `#!/bin/sh
if [ "$1" = auth ] && [ "$2" = credential ]; then
  printf '%s\n' '{"version":1,"base_url":"https://api.everyapi.ai/v1","api_key":"from-login"}'
  exit 0
fi
exit 1
`)
	setLLMEnv(t, "", "", "", "")
	tools := ToolsFromEnv()
	if tools.Translator.APIKey != "" || tools.Translator.Credential == nil {
		t.Fatalf("translator = %#v", tools.Translator)
	}
	got, err := tools.Translator.Credential(context.Background(), false)
	if err != nil {
		t.Fatal(err)
	}
	if got != (Endpoint{BaseURL: EveryAPIBaseURL, APIKey: "from-login"}) {
		t.Fatalf("endpoint = %#v", got)
	}
	if err := tools.Check(); err != nil && strings.Contains(err.Error(), "EveryAPI") {
		t.Fatal(err)
	}
}

func TestToolsFromEnvReportsSignedOutCLI(t *testing.T) {
	writeFakeEveryAPI(t, `#!/bin/sh
echo 'EVERYAPI_CREDENTIAL_ERROR:not_logged_in: not logged in' >&2
exit 1
`)
	setLLMEnv(t, "", "", "", "")
	tools := ToolsFromEnv()
	_, err := tools.Translator.Credential(context.Background(), false)
	if err == nil || !strings.Contains(err.Error(), "everyapi auth login") {
		t.Fatalf("err = %v", err)
	}
}

func TestToolsTranscribeRejectsEmptyVideoPath(t *testing.T) {
	_, err := Tools{FFmpeg: "binary-that-does-not-exist", Whisper: "binary-that-does-not-exist"}.Transcribe(context.Background(), " ")
	if err == nil || !strings.Contains(err.Error(), "video path is empty") {
		t.Fatalf("err = %v", err)
	}
}

func TestWhisperThreadCountUsesAvailableCPU(t *testing.T) {
	want := runtime.NumCPU()
	if want < 1 {
		want = 1
	}
	if want > 8 {
		want = 8
	}
	if got := whisperThreadCount(); got != want {
		t.Fatalf("threads = %d, want %d", got, want)
	}
}

func TestToolsDownloadReturnsLocalFileWithoutYTDLP(t *testing.T) {
	video := filepath.Join(t.TempDir(), "talk.mp4")
	if err := os.WriteFile(video, []byte("video"), 0o644); err != nil {
		t.Fatal(err)
	}
	url := "file://" + video
	got, err := (Tools{}).Download(context.Background(), url)
	if err != nil {
		t.Fatal(err)
	}
	if got != video {
		t.Fatalf("path = %q, want %q", got, video)
	}
}

func TestToolsDownloadUnescapesLocalFileURL(t *testing.T) {
	video := filepath.Join(t.TempDir(), "访谈 01.mp4")
	if err := os.WriteFile(video, []byte("video"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := (Tools{}).Download(context.Background(), (&url.URL{Scheme: "file", Path: video}).String())
	if err != nil {
		t.Fatal(err)
	}
	if got != video {
		t.Fatalf("path = %q, want %q", got, video)
	}
}

func TestLocalVideoPathRejectsRemoteFileHost(t *testing.T) {
	if path, ok := localVideoPath("file://other-machine/video.mp4"); ok || path != "" {
		t.Fatalf("path = %q, ok = %v", path, ok)
	}
}

func TestLastLineAndTail(t *testing.T) {
	if got := lastLine("[download] 100%\n/videos/a.mp4\n\n"); got != "/videos/a.mp4" {
		t.Fatalf("lastLine = %q", got)
	}
	if got := tail("abcdef", 3); got != "...def" {
		t.Fatalf("tail = %q", got)
	}
	if got := tail(" short ", 10); got != "short" {
		t.Fatalf("tail = %q", got)
	}
}

// fakeYTDLP writes a yt-dlp that prints the captured stdout and stderr of a real run, then exits with code.
func fakeYTDLP(t *testing.T, stdout, stderr string, code int) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "yt-dlp")
	script := "#!/bin/sh\ncat " + shellQuote(stdout) + "\ncat " + shellQuote(stderr) + " >&2\nexit " + strconv.Itoa(code) + "\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func shellQuote(path string) string {
	if path == "" {
		return "/dev/null"
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		panic(err)
	}
	return "'" + strings.ReplaceAll(abs, "'", `'\''`) + "'"
}

func TestDownloadFollowsTheRealOutput(t *testing.T) {
	tools := Tools{YTDLP: fakeYTDLP(t, "testdata/ytdlp/download-youtube.stdout", "", 0), OutputDir: t.TempDir()}
	var steps []core.Progress
	var info Info
	ctx := WithTrace(context.Background(), &Trace{Progress: func(p core.Progress) { steps = append(steps, p) }, Info: func(i Info) { info = i }})
	path, err := tools.Download(ctx, "https://www.youtube.com/watch?v=jNQXAC9IVRw")
	if err != nil {
		t.Fatal(err)
	}
	if path != "/library/Me at the zoo [jNQXAC9IVRw]/video.mp4" || info.Title != "Me at the zoo" || len(steps) == 0 || steps[len(steps)-1].Fraction != 1 {
		t.Fatalf("path %q, info %+v, %d steps", path, info, len(steps))
	}
}

func TestDownloadNamesAFailedYouTubeVideoFromOEmbed(t *testing.T) {
	var asked string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked = r.URL.Query().Get("url")
		w.Write([]byte(`{"title":"Me at the zoo","thumbnail_url":"https://i.ytimg.com/vi/jNQXAC9IVRw/hqdefault.jpg"}`))
	}))
	t.Cleanup(srv.Close)
	tools := Tools{YTDLP: fakeYTDLP(t, "", "testdata/ytdlp/errors/outdated-youtube.stderr", 1), OutputDir: t.TempDir(), OEmbed: srv.URL}
	var info Info
	ctx := WithTrace(context.Background(), &Trace{Info: func(i Info) { info = i }})
	_, err := tools.Download(ctx, "https://youtu.be/jNQXAC9IVRw")
	if err == nil || !strings.Contains(err.Error(), "No video formats found") {
		t.Fatalf("err = %v", err)
	}
	if asked != "https://www.youtube.com/watch?v=jNQXAC9IVRw" || info.Title != "Me at the zoo" || info.Thumbnail == "" {
		t.Fatalf("asked %q, info %+v", asked, info)
	}
}

func TestDownloadDoesNotAskOEmbedForOtherSites(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("asked oEmbed for %s", r.URL)
	}))
	t.Cleanup(srv.Close)
	tools := Tools{YTDLP: fakeYTDLP(t, "", "testdata/ytdlp/errors/unsupported.stderr", 1), OutputDir: t.TempDir(), OEmbed: srv.URL}
	if _, err := tools.Download(context.Background(), "https://example.com/"); err == nil {
		t.Fatal("expected an error")
	}
}

// argsYTDLP writes a yt-dlp that saves its arguments to a file, one a line, and fails.
func argsYTDLP(t *testing.T) (tool, args string) {
	t.Helper()
	dir := t.TempDir()
	tool, args = filepath.Join(dir, "yt-dlp"), filepath.Join(dir, "args")
	if err := os.WriteFile(tool, []byte("#!/bin/sh\nfor a in \"$@\"; do echo \"$a\"; done > "+shellQuote(args)+"\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return tool, args
}

func TestDownloadBorrowsTheBrowsersLogin(t *testing.T) {
	tool, args := argsYTDLP(t)
	tools := Tools{YTDLP: tool, OutputDir: t.TempDir(), Cookies: "chrome"}
	tools.Download(context.Background(), "https://www.bilibili.com/video/BV1xx411c7mD")
	data, _ := os.ReadFile(args)
	if !strings.Contains(string(data), "--cookies-from-browser\nchrome\n") {
		t.Fatalf("args %s", data)
	}
	tools.Cookies = ""
	tools.Download(context.Background(), "https://www.bilibili.com/video/BV1xx411c7mD")
	data, _ = os.ReadFile(args)
	if strings.Contains(string(data), "--cookies-from-browser") {
		t.Fatalf("args without cookies %s", data)
	}
}

// An app opened from Finder has a PATH without Homebrew's folder, where yt-dlp would look for ffmpeg to merge the video with its sound; Download names the ffmpeg Kando found.
func TestDownloadTellsYTDLPWhereFFmpegIs(t *testing.T) {
	tool, args := argsYTDLP(t)
	tools := Tools{YTDLP: tool, FFmpeg: "/opt/homebrew/bin/ffmpeg", OutputDir: t.TempDir()}
	tools.Download(context.Background(), "https://www.youtube.com/watch?v=jNQXAC9IVRw")
	data, _ := os.ReadFile(args)
	if !strings.Contains(string(data), "--ffmpeg-location\n/opt/homebrew/bin/ffmpeg\n") {
		t.Fatalf("args %s", data)
	}
}

// yt-dlp keeps a split-format video's separate tracks and its .part files in a folder of its own outside the library, so the library watcher only ever sees the merged video; the folder is the same for the same video, so a retry picks up where the last try stopped.
func TestDownloadKeepsPartialFilesOutsideTheLibrary(t *testing.T) {
	tool, args := argsYTDLP(t)
	library, cache := t.TempDir(), t.TempDir()
	tools := Tools{YTDLP: tool, OutputDir: library, DownloadTemp: cache}
	temps := map[string]string{}
	for _, source := range []string{"https://www.youtube.com/watch?v=jNQXAC9IVRw", "https://www.youtube.com/watch?v=jNQXAC9IVRw", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"} {
		tools.Download(context.Background(), source)
		home, temp := ytdlpPaths(t, args)
		if home != library || !filepath.IsAbs(temp) || !pathInside(cache, temp) {
			t.Fatalf("home %q temp %q", home, temp)
		}
		if seen, ok := temps[source]; ok && seen != temp {
			t.Fatalf("%s: temp %q then %q", source, seen, temp)
		}
		temps[source] = temp
	}
	if temps["https://www.youtube.com/watch?v=jNQXAC9IVRw"] == temps["https://www.youtube.com/watch?v=dQw4w9WgXcQ"] {
		t.Fatalf("two videos share a temp folder: %v", temps)
	}
}

// ytdlpPaths reads the home path (-P) and the temp path (-P temp:) from the arguments argsYTDLP saved.
func ytdlpPaths(t *testing.T, args string) (home, temp string) {
	t.Helper()
	data, err := os.ReadFile(args)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(string(data), "\n")
	for i := 0; i+1 < len(lines); i++ {
		if lines[i] != "-P" {
			continue
		}
		if rest, ok := strings.CutPrefix(lines[i+1], "temp:"); ok {
			temp = rest
		} else {
			home = lines[i+1]
		}
	}
	return home, temp
}

// yt-dlp moves the merged video into the library with a rename only when its temp path is on the library's volume; across volumes it copies the video into video.mp4 in place, so a download stopped during the copy leaves a cut-off video.mp4 that the next try takes for a finished one. When DownloadTemp is not on the library's volume, or cannot be told to be, the partial files go to a hidden folder in the library, which the library never lists, and the folder is gone once the video is in.
func TestDownloadKeepsPartialFilesOnTheLibrarysVolume(t *testing.T) {
	notAFolder := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(notAFolder, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	tool, args := argsYTDLP(t)
	library := t.TempDir()
	tools := Tools{YTDLP: tool, OutputDir: library, DownloadTemp: filepath.Join(notAFolder, "downloads")}
	tools.Download(context.Background(), "https://www.youtube.com/watch?v=jNQXAC9IVRw")
	if _, temp := ytdlpPaths(t, args); !pathInside(filepath.Join(library, ".kando-downloads"), temp) {
		t.Fatalf("temp %q", temp)
	}
	tools.YTDLP, _ = splitYTDLP(t)
	path, err := tools.Download(context.Background(), "https://www.youtube.com/watch?v=jNQXAC9IVRw")
	if err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(path); string(data) != "track 395track 251" {
		t.Fatalf("%s = %q", path, data)
	}
	if entries, _ := os.ReadDir(library); len(entries) != 1 || entries[0].Name() != filepath.Base(filepath.Dir(path)) {
		t.Fatalf("library holds %v", entries)
	}
}

// splitYTDLP writes a yt-dlp that downloads a split-format video as the real one does: the video and sound tracks go to the temp path (-P temp:) or, without one, next to the video, and are merged into <home>/Me at the zoo [jNQXAC9IVRw]/video.mp4. When that file is already there it is reported as downloaded and left as it is. Each run appends a line to the returned file.
func splitYTDLP(t *testing.T) (tool, runs string) {
	t.Helper()
	dir := t.TempDir()
	tool, runs = filepath.Join(dir, "yt-dlp"), filepath.Join(dir, "runs")
	script := `#!/bin/sh
home=; temp=
while [ $# -gt 0 ]; do
	case "$1" in
	-P) shift; case "$1" in temp:*) temp="${1#temp:}" ;; *) home="$1" ;; esac ;;
	--) break ;;
	esac
	shift
done
echo run >> ` + shellQuote(runs) + `
folder="Me at the zoo [jNQXAC9IVRw]"
final="$home/$folder/video.mp4"
if [ -e "$final" ]; then echo "$final"; exit 0; fi
work="${temp:-$home}/$folder"
mkdir -p "$work" "$home/$folder" || exit 1
printf 'track 395' > "$work/video.f395.mp4"
printf 'track 251' > "$work/video.f251.webm"
cat "$work/video.f395.mp4" "$work/video.f251.webm" > "$work/video.temp.mp4" || exit 1
rm "$work/video.f395.mp4" "$work/video.f251.webm"
mv "$work/video.temp.mp4" "$final" || exit 1
echo "$final"
`
	if err := os.WriteFile(tool, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return tool, runs
}

func TestDownloadLeavesNoTemporaryFolderBehind(t *testing.T) {
	tool, _ := splitYTDLP(t)
	library, cache := t.TempDir(), t.TempDir()
	tools := Tools{YTDLP: tool, OutputDir: library, DownloadTemp: cache}
	path, err := tools.Download(context.Background(), "https://www.youtube.com/watch?v=jNQXAC9IVRw")
	if err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(path); string(data) != "track 395track 251" {
		t.Fatalf("%s = %q", path, data)
	}
	if entries, _ := os.ReadDir(cache); len(entries) != 0 {
		t.Fatalf("temp folders left: %v", entries)
	}
	if entries, _ := os.ReadDir(filepath.Dir(path)); len(entries) != 1 {
		t.Fatalf("library folder holds %v", entries)
	}
}

// An earlier Kando renamed the video track of an unfinished download to video.mp4 and left the sound track beside it. yt-dlp takes that silent file for a finished download, so a retry would transcribe it forever; Download notices the leftover track, removes both and downloads the video again.
func TestDownloadReplacesAVideoLeftWithoutItsSound(t *testing.T) {
	tool, runs := splitYTDLP(t)
	library := t.TempDir()
	folder := filepath.Join(library, "Me at the zoo [jNQXAC9IVRw]")
	if err := os.Mkdir(folder, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string]string{"video.mp4": "track 395", "video.f251.webm": "track 251"} {
		if err := os.WriteFile(filepath.Join(folder, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	tools := Tools{YTDLP: tool, OutputDir: library, DownloadTemp: t.TempDir()}
	path, err := tools.Download(context.Background(), "https://www.youtube.com/watch?v=jNQXAC9IVRw")
	if err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(path); path != filepath.Join(folder, "video.mp4") || string(data) != "track 395track 251" {
		t.Fatalf("%s = %q", path, data)
	}
	if _, err := os.Stat(filepath.Join(folder, "video.f251.webm")); !os.IsNotExist(err) {
		t.Fatalf("leftover track: %v", err)
	}
	if data, _ := os.ReadFile(runs); strings.Count(string(data), "run") != 2 {
		t.Fatalf("yt-dlp ran %q", data)
	}
}

// A video downloaded before, with nothing unfinished beside it, is used as it is.
func TestDownloadReusesAFinishedVideo(t *testing.T) {
	tool, runs := splitYTDLP(t)
	library := t.TempDir()
	folder := filepath.Join(library, "Me at the zoo [jNQXAC9IVRw]")
	if err := os.Mkdir(folder, 0o755); err != nil {
		t.Fatal(err)
	}
	video := filepath.Join(folder, "video.mp4")
	if err := os.WriteFile(video, []byte("finished"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(folder, "video.bilingual.srt"), []byte("1"), 0o644); err != nil {
		t.Fatal(err)
	}
	tools := Tools{YTDLP: tool, OutputDir: library, DownloadTemp: t.TempDir()}
	path, err := tools.Download(context.Background(), "https://www.youtube.com/watch?v=jNQXAC9IVRw")
	if err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(path); path != video || string(data) != "finished" {
		t.Fatalf("%s = %q", path, data)
	}
	if data, _ := os.ReadFile(runs); strings.Count(string(data), "run") != 1 {
		t.Fatalf("yt-dlp ran %q", data)
	}
}
