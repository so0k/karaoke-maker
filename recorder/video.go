package recorder

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// VideoGenerator handles FFmpeg integration for creating MP4 from frames
type VideoGenerator struct {
	framesDir  string
	audioPath  string // optional; empty means no audio
	outputPath string
	frameRate  float64
	ffmpegPath string
}

// NewVideoGenerator creates a new video generator
func NewVideoGenerator(framesDir, audioPath, outputPath string) *VideoGenerator {
	return &VideoGenerator{
		framesDir:  framesDir,
		audioPath:  audioPath,
		outputPath: outputPath,
		frameRate:  30.0, // default 30 FPS
		ffmpegPath: findFFmpegPath(),
	}
}

// findFFmpegPath locates ffmpeg binary
func findFFmpegPath() string {
	// Try common locations
	paths := []string{
		"/Users/vincentdesmet/.local/share/mise/installs/ffmpeg/7.1.1/bin/ffmpeg",
		"/usr/local/bin/ffmpeg",
		"/opt/homebrew/bin/ffmpeg",
		"ffmpeg", // system PATH
	}

	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	// Fall back to PATH lookup
	if path, err := exec.LookPath("ffmpeg"); err == nil {
		return path
	}

	return "ffmpeg" // assume it's in PATH
}

// GenerateVideo creates MP4 video from frame sequence and audio
func (vg *VideoGenerator) GenerateVideo() error {
	// Ensure output directory exists
	outputDir := filepath.Dir(vg.outputPath)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Build ffmpeg command
	framePattern := filepath.Join(vg.framesDir, "frame_%04d.png")

	args := []string{"-y", "-framerate", fmt.Sprintf("%.6f", vg.frameRate), "-i", framePattern}
	if vg.audioPath != "" {
		args = append(args, "-i", vg.audioPath, "-map", "0:v", "-map", "1:a")
	}
	// Video codec and pixel format
	args = append(args, "-c:v", "h264_videotoolbox", "-pix_fmt", "yuv420p")
	if vg.audioPath != "" {
		args = append(args, "-c:a", "aac", "-shortest")
	}
	args = append(args, vg.outputPath)

	fmt.Printf("Running FFmpeg command: %s %v\n", vg.ffmpegPath, args)

	// Execute ffmpeg command
	cmd := exec.Command(vg.ffmpegPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg failed: %w", err)
	}

	fmt.Printf("Video generated successfully: %s\n", vg.outputPath)
	return nil
}

// SetFrameRate configures the output frame rate
func (vg *VideoGenerator) SetFrameRate(fps float64) {
	vg.frameRate = fps
}

// SetFFmpegPath sets a custom ffmpeg binary path
func (vg *VideoGenerator) SetFFmpegPath(path string) {
	vg.ffmpegPath = path
}

// CleanupFrames removes temporary frame files
func (vg *VideoGenerator) CleanupFrames() error {
	if vg.framesDir == "" {
		return nil
	}

	// Remove all PNG files in frames directory
	pattern := filepath.Join(vg.framesDir, "*.png")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("failed to glob frame files: %w", err)
	}

	for _, match := range matches {
		if err := os.Remove(match); err != nil {
			fmt.Printf("Warning: failed to remove frame file %s: %v\n", match, err)
		}
	}

	// Remove frames directory if it's empty
	if err := os.Remove(vg.framesDir); err != nil && !os.IsNotExist(err) {
		fmt.Printf("Warning: failed to remove frames directory %s: %v\n", vg.framesDir, err)
	}

	fmt.Printf("Cleaned up %d frame files\n", len(matches))
	return nil
}

// ValidateInputs checks that required input files exist
func (vg *VideoGenerator) ValidateInputs() error {
	// Check if frames directory exists
	if _, err := os.Stat(vg.framesDir); os.IsNotExist(err) {
		return fmt.Errorf("frames directory does not exist: %s", vg.framesDir)
	}

	// Check if audio file exists (optional)
	if vg.audioPath != "" {
		if _, err := os.Stat(vg.audioPath); os.IsNotExist(err) {
			return fmt.Errorf("audio file does not exist: %s", vg.audioPath)
		}
	}

	// Check if frames exist
	pattern := filepath.Join(vg.framesDir, "frame_*.png")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("failed to check for frame files: %w", err)
	}

	if len(matches) == 0 {
		return fmt.Errorf("no frame files found in: %s", vg.framesDir)
	}

	fmt.Printf("Found %d frame files for video generation\n", len(matches))
	return nil
}

// GetOutputPath returns the configured output path
func (vg *VideoGenerator) GetOutputPath() string {
	return vg.outputPath
}
