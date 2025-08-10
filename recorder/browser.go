package recorder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/rod/lib/proto"
)

// BrowserRecorder handles browser automation for capturing ttyd screenshots
type BrowserRecorder struct {
	browser    *rod.Browser
	page       *rod.Page
	framesDir  string
	frameCount int
}

// NewBrowserRecorder creates a new browser recorder
func NewBrowserRecorder(framesDir string) *BrowserRecorder {
	return &BrowserRecorder{
		framesDir: framesDir,
	}
}

// Start initializes the browser and connects to ttyd
func (br *BrowserRecorder) Start(ttydURL string) error {
	// Launch browser in headless mode
	path, exists := launcher.LookPath()
	if !exists {
		return fmt.Errorf("chrome/chromium browser not found")
	}

	launcher := launcher.New().Bin(path).Headless(true)
	url, err := launcher.Launch()
	if err != nil {
		return fmt.Errorf("failed to launch browser: %w", err)
	}

	// Connect to browser
	br.browser = rod.New().ControlURL(url)
	if err := br.browser.Connect(); err != nil {
		return fmt.Errorf("failed to connect to browser: %w", err)
	}

	// Create new page with target URL (VHS approach)
	br.page, err = br.browser.Page(proto.TargetCreateTarget{
		URL: ttydURL,
	})
	if err != nil {
		return fmt.Errorf("failed to create browser page: %w", err)
	}

	// Wait for ttyd to load
	if err := br.page.WaitLoad(); err != nil {
		return fmt.Errorf("failed to wait for ttyd page load: %w", err)
	}

	// Wait for terminal to be ready
	time.Sleep(2 * time.Second)

	fmt.Printf("🌐 Browser connected to ttyd at %s\n", ttydURL)
	return nil
}

// CaptureFrame takes a screenshot and saves it as a frame
func (br *BrowserRecorder) CaptureFrame() (string, error) {
	// Find xterm.js text layer canvas (VHS approach)
	textCanvas, err := br.page.Element("canvas.xterm-text-layer")
	if err != nil {
		// Fallback to any canvas element
		textCanvas, err = br.page.Element("canvas")
		if err != nil {
			return "", fmt.Errorf("failed to find terminal canvas: %w", err)
		}
	}

	// Screenshot the terminal canvas element (not full viewport)
	screenshot, err := textCanvas.Screenshot(proto.PageCaptureScreenshotFormatPng, 0)
	if err != nil {
		return "", fmt.Errorf("failed to capture terminal screenshot: %w", err)
	}

	// Generate filename
	br.frameCount++
	filename := fmt.Sprintf("frame_%04d.png", br.frameCount)
	filepath := filepath.Join(br.framesDir, filename)

	// Save screenshot
	if err := os.WriteFile(filepath, screenshot, 0644); err != nil {
		return "", fmt.Errorf("failed to save screenshot: %w", err)
	}

	return filepath, nil
}

// CaptureFrameRaw screenshots the terminal canvas but returns the PNG bytes
// without writing them to disk or incrementing the frame counter. Useful for
// calibration and timing measurements without filesystem overhead.
func (br *BrowserRecorder) CaptureFrameRaw() ([]byte, error) {
	if br.page == nil {
		return nil, fmt.Errorf("browser page not initialized")
	}

	// Find xterm.js text layer canvas (VHS approach)
	textCanvas, err := br.page.Element("canvas.xterm-text-layer")
	if err != nil {
		// Fallback to any canvas element
		textCanvas, err = br.page.Element("canvas")
		if err != nil {
			return nil, fmt.Errorf("failed to find terminal canvas: %w", err)
		}
	}

	// Screenshot element to PNG bytes
	screenshot, err := textCanvas.Screenshot(proto.PageCaptureScreenshotFormatPng, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to capture terminal screenshot: %w", err)
	}
	return screenshot, nil
}

// CaptureScreenshotPNG captures the terminal canvas and returns PNG bytes.
// Alias of CaptureFrameRaw for clarity in call sites.
func (br *BrowserRecorder) CaptureScreenshotPNG() ([]byte, error) {
    return br.CaptureFrameRaw()
}

// SetViewportSize configures the browser viewport size
func (br *BrowserRecorder) SetViewportSize(width, height int) error {
	if br.page == nil {
		return fmt.Errorf("browser page not initialized")
	}

	return br.page.SetViewport(&proto.EmulationSetDeviceMetricsOverride{
		Width:  width,
		Height: height,
	})
}

// WaitForTerminalReady waits for the terminal to be fully loaded and ready
func (br *BrowserRecorder) WaitForTerminalReady() error {
	if br.page == nil {
		return fmt.Errorf("browser page not initialized")
	}

	// Wait for xterm.js elements specifically (VHS approach)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Look for xterm.js specific selectors (what VHS looks for)
	selectors := []string{
		"canvas.xterm-text-layer",   // xterm.js text layer (primary target)
		"canvas.xterm-cursor-layer", // xterm.js cursor layer
		".xterm-viewport",           // xterm viewport
		"canvas",                    // fallback to any canvas
	}

	for _, selector := range selectors {
		element, err := br.page.Context(ctx).Element(selector)
		if err == nil && element != nil {
			fmt.Printf("📺 Terminal element found: %s\n", selector)
			// Additional wait for terminal to fully render content
			time.Sleep(3 * time.Second)
			return nil
		}
	}

	return fmt.Errorf("terminal canvas not found after 15 seconds")
}

// SendCommand sends a command to the terminal (VHS approach)
func (br *BrowserRecorder) SendCommand(command string) error {
	if br.page == nil {
		return fmt.Errorf("browser page not initialized")
	}

	// Find the textarea element where VHS sends keystrokes
	textarea, err := br.page.Element("textarea")
	if err != nil {
		return fmt.Errorf("failed to find textarea: %w", err)
	}

	// Focus on the textarea first
	if err := textarea.Focus(); err != nil {
		return fmt.Errorf("failed to focus textarea: %w", err)
	}

	// Type the command directly into the textarea (simpler approach)
	if err := textarea.Input(command); err != nil {
		return fmt.Errorf("failed to input command: %w", err)
	}

	// Send Enter key using JavaScript event dispatch (most reliable)
	_, err = br.page.Eval(`() => {
		const textarea = document.querySelector('textarea');
		if (textarea) {
			textarea.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', keyCode: 13}));
		}
	}`)
	if err != nil {
		return fmt.Errorf("failed to send Enter key: %w", err)
	}

	fmt.Printf("⌨️  Sent command: %s\n", command)
	return nil
}

// GetFrameCount returns the current frame count
func (br *BrowserRecorder) GetFrameCount() int {
	return br.frameCount
}

// Close stops the browser
func (br *BrowserRecorder) Close() error {
	if br.browser != nil {
		err := br.browser.Close()
		if err != nil {
			return fmt.Errorf("failed to close browser: %w", err)
		}
		fmt.Println("🌐 Browser closed")
	}
	return nil
}
