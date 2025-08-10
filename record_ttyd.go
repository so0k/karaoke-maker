//go:build ignore
// +build ignore

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"karaoke-tui/recorder"
)

const (
	// Recording configuration
	FrameRate      = 30   // FPS
	ViewportWidth  = 1440 // Browser viewport width
	ViewportHeight = 900  // Browser viewport height
)

func main() {
	if err := runTTYDRecording(); err != nil {
		log.Fatalf("Recording failed: %v", err)
	}
}

func runTTYDRecording() error {
	fmt.Println("🎬 Starting karaoke TUI recording with ttyd approach...")

	// Setup paths
	binaryPath := "./karaoke-tui"
	audioPath := "audio/from_boilerplate_to_flow.mp3"
	timingPath := "audio/from_boilerplate_to_flow_timings.json"
	framesDir := "frames"
	outputPath := "karaoke_output.mp4"

	// Get recording duration from timing metadata
	recordingDuration, err := getRecordingDuration(timingPath)
	if err != nil {
		return fmt.Errorf("failed to get recording duration: %w", err)
	}
	fmt.Printf("📏 Detected song duration: %.1f seconds\n", recordingDuration.Seconds())

	// Validate prerequisites
	if err := validatePrerequisites(binaryPath, audioPath); err != nil {
		return fmt.Errorf("prerequisites check failed: %w", err)
	}

	// Create frames directory
	if err := os.MkdirAll(framesDir, 0755); err != nil {
		return fmt.Errorf("failed to create frames directory: %w", err)
	}

	// Start ttyd server with shell (VHS approach)
	fmt.Println("🚀 Starting ttyd server with shell...")
	ttydServer, err := recorder.NewTTYDServer()
	if err != nil {
		return fmt.Errorf("failed to create ttyd server: %w", err)
	}
	defer ttydServer.Stop()

	if err := ttydServer.Start(); err != nil {
		return fmt.Errorf("failed to start ttyd server: %w", err)
	}

	// Initialize browser recorder
	browserRecorder := recorder.NewBrowserRecorder(framesDir)
	defer browserRecorder.Close()

	// Connect browser to ttyd
	fmt.Println("🌐 Connecting browser to ttyd...")
	if err := browserRecorder.Start(ttydServer.GetURL()); err != nil {
		return fmt.Errorf("failed to start browser recorder: %w", err)
	}

	// Set viewport size
	if err := browserRecorder.SetViewportSize(ViewportWidth, ViewportHeight); err != nil {
		return fmt.Errorf("failed to set viewport size: %w", err)
	}

	// Wait for terminal to be ready
	if err := browserRecorder.WaitForTerminalReady(); err != nil {
		return fmt.Errorf("failed to wait for terminal: %w", err)
	}

	// Fixed timescale approach for deterministic capture & simpler debugging
	timescale := 3.0
	fmt.Printf("🧮 Using fixed timescale: %.2fx (target wall/frame ≈ 100ms)\n", timescale)
	// Optional calibration as a warning if machine is too slow for 100ms/frame
	fmt.Println("🧪 Quick check: measuring capture p95...")
	if p95Dur, avgDur, err := calibrateCaptureFull(browserRecorder, 60); err == nil {
		p95ms := float64(p95Dur.Microseconds()) / 1000.0
		avgms := float64(avgDur.Microseconds()) / 1000.0
		fmt.Printf("⏱️  Avg: %.1f ms  P95: %.1f ms\n", avgms, p95ms)
		if p95ms > 100.0 {
			fmt.Printf("⚠️  P95 > 100ms; capture may overrun. Close apps or lower resolution.\n")
		}
	} else {
		fmt.Printf("(calibration skipped due to error)\n")
	}

	// Send command to start karaoke TUI with fixed timescale
	command := "./karaoke-tui -timescale " + strconv.FormatFloat(timescale, 'f', 2, 64)
	if timescale == 1.0 {
		// You can enable audio if you want live verification
		// command += " -audio"
	}
	if err := browserRecorder.SendCommand(command); err != nil {
		return fmt.Errorf("failed to send command: %w", err)
	}

	// Small delay for command to execute and TUI to initialize
	fmt.Println("⏰ Waiting for TUI to start...")
	time.Sleep(1 * time.Second)

	// Recording loop
	fmt.Printf("⏺️  Recording for %v at %d FPS...\n", recordingDuration, FrameRate)

	// Deterministic scheduler: capture exactly N frames at 30fps content.
	// With timescale=3.0, interval is 100ms wall per frame (10fps wall → 30fps content).
	startTime := time.Now()
	N := int(recordingDuration.Seconds()*float64(FrameRate) + 0.5)
	interval := 100 * time.Millisecond
	if timescale != 3.0 {
		interval = time.Duration(float64(time.Second) * (timescale / float64(FrameRate)))
	}
	overruns := 0
	const overrunWarn = 15 * time.Millisecond
	frameCount := 0

	for i := 0; i < N; i++ {
		target := startTime.Add(time.Duration(i) * interval)
		if d := time.Until(target); d > 0 {
			time.Sleep(d)
		} else if -d > overrunWarn {
			overruns++
		}
		if _, err := browserRecorder.CaptureFrame(); err != nil {
			fmt.Printf("Warning: failed to capture frame %d: %v\n", i+1, err)
		}
		frameCount++
		if frameCount%300 == 0 {
			elapsed := time.Since(startTime)
			fmt.Printf("📹 Captured %d/%d frames (%.1fs elapsed)\n", frameCount, N, elapsed.Seconds())
		}
	}
	if overruns > 0 {
		fmt.Printf("⚠️  Detected %d scheduling overruns (>%.0fms late). Consider increasing timescale.\n", overruns, overrunWarn.Seconds()*1000)
	}

	fmt.Printf("🎞️  Captured %d frames total\n", frameCount)

	// Generate video
	fmt.Println("🎥 Generating MP4 video...")
	// Always include audio when muxing so we get a soundtrack, even if
	// timescaled (ffmpeg -shortest will clip to video length).
	videoGen := recorder.NewVideoGenerator(framesDir, audioPath, outputPath)

	if err := videoGen.ValidateInputs(); err != nil {
		return fmt.Errorf("video generation validation failed: %w", err)
	}
	// Always encode at 30fps (exactly N=duration*30 frames were captured)
	fmt.Printf("🧮 Using fixed input framerate: %d FPS (frames=%d, duration=%.3fs)\n", FrameRate, frameCount, recordingDuration.Seconds())
	videoGen.SetFrameRate(float64(FrameRate))
	if err := videoGen.GenerateVideo(); err != nil {
		return fmt.Errorf("video generation failed: %w", err)
	}

	// Cleanup frames (commented out for debugging)
	fmt.Println("🧹 Skipping frame cleanup for debugging - frames kept in:", framesDir)
	// if err := videoGen.CleanupFrames(); err != nil {
	// 	fmt.Printf("Warning: cleanup failed: %v\n", err)
	// }

	fmt.Printf("🎉 Recording complete! Output: %s\n", outputPath)

	// Print file info
	if info, err := os.Stat(outputPath); err == nil {
		fmt.Printf("📊 Video file size: %.1f MB\n", float64(info.Size())/(1024*1024))
	}

	return nil
}

// TimingMeta represents the metadata section of the timing JSON
type TimingMeta struct {
	Duration int `json:"duration"`
}

// TimingData represents the timing JSON structure
type TimingData struct {
	Meta TimingMeta `json:"meta"`
}

// getRecordingDuration reads the duration from the timing JSON file
func getRecordingDuration(timingPath string) (time.Duration, error) {
	data, err := os.ReadFile(timingPath)
	if err != nil {
		return 0, fmt.Errorf("failed to read timing file: %w", err)
	}

	var timing TimingData
	if err := json.Unmarshal(data, &timing); err != nil {
		return 0, fmt.Errorf("failed to parse timing JSON: %w", err)
	}

	if timing.Meta.Duration <= 0 {
		return 0, fmt.Errorf("invalid duration in timing metadata: %d", timing.Meta.Duration)
	}

	return time.Duration(timing.Meta.Duration) * time.Second, nil
}

func validatePrerequisites(binaryPath, audioPath string) error {
	// Check if karaoke TUI binary exists
	if _, err := os.Stat(binaryPath); os.IsNotExist(err) {
		pwd, _ := os.Getwd()
		fmt.Printf("❌ Prerequisite check failed PWD: %s\n", pwd)
		return fmt.Errorf("karaoke TUI binary not found: %s\nRun 'go build -o karaoke-tui .' first", binaryPath)
	}

	// Check if audio file exists
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		return fmt.Errorf("audio file not found: %s", audioPath)
	}

	// Check if binary is executable
	if info, err := os.Stat(binaryPath); err == nil {
		if info.Mode()&0111 == 0 {
			return fmt.Errorf("karaoke TUI binary is not executable: %s", binaryPath)
		}
	}

	// Check if ttyd is available
	if _, err := os.Stat("/opt/homebrew/bin/ttyd"); os.IsNotExist(err) {
		return fmt.Errorf("ttyd not found at /opt/homebrew/bin/ttyd")
	}

	fmt.Println("✅ Prerequisites validated")
	return nil
}

func init() {
	// Ensure we're in the right directory
	if _, err := os.Stat("main.go"); os.IsNotExist(err) {
		// Try to find the project directory
		if dir := os.Getenv("PWD"); dir != "" && strings.Contains(dir, "karaoke-maker") {
			os.Chdir(dir)
		}
	}
}

// calibrateCapture measures average CaptureFrame time using raw capture
// without disk I/O to estimate a safe timescale for 30fps recording.
func calibrateCapture(br *recorder.BrowserRecorder, samples int) (time.Duration, error) {
	if samples <= 0 {
		samples = 20
	}
	// Warm-up
	if _, err := br.CaptureFrameRaw(); err != nil {
		return 0, err
	}
	// Measure
	var total time.Duration
	for i := 0; i < samples; i++ {
		t0 := time.Now()
		if _, err := br.CaptureFrameRaw(); err != nil {
			return 0, err
		}
		total += time.Since(t0)
		// tiny pause to avoid hammering
		time.Sleep(5 * time.Millisecond)
	}
	return time.Duration(int64(total) / int64(samples)), nil
}

// calibrateCaptureFull measures end-to-end screenshot + disk write to better
// approximate real per-frame cost. Returns (p95, average).
func calibrateCaptureFull(br *recorder.BrowserRecorder, samples int) (time.Duration, time.Duration, error) {
	if samples <= 0 {
		samples = 60
	}
	// Warm-up
	if _, err := br.CaptureScreenshotPNG(); err != nil {
		return 0, 0, err
	}
	// Ensure temp dir exists
	tmpDir := os.TempDir()
	durs := make([]time.Duration, 0, samples)
	var total time.Duration
	for i := 0; i < samples; i++ {
		t0 := time.Now()
		png, err := br.CaptureScreenshotPNG()
		if err != nil {
			return 0, 0, err
		}
		// Write to temp file to include disk I/O
		name := fmt.Sprintf("tty_calib_%d_%d.png", time.Now().UnixNano(), i)
		path := tmpDir + string(os.PathSeparator) + name
		if err := os.WriteFile(path, png, 0644); err != nil {
			return 0, 0, err
		}
		// Remove file
		_ = os.Remove(path)
		dur := time.Since(t0)
		durs = append(durs, dur)
		total += dur
		time.Sleep(5 * time.Millisecond)
	}
	// Compute p95
	sort.Slice(durs, func(i, j int) bool { return durs[i] < durs[j] })
	idx := int(float64(len(durs))*0.95 + 0.5)
	if idx >= len(durs) {
		idx = len(durs) - 1
	} else if idx < 0 {
		idx = 0
	}
	p95 := durs[idx]
	avg := time.Duration(int64(total) / int64(len(durs)))
	return p95, avg, nil
}
