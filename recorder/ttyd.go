package recorder

import (
	"fmt"
	"net"
	"os/exec"
	"time"
)

// TTYDServer manages a ttyd server instance for terminal rendering
type TTYDServer struct {
	cmd     *exec.Cmd
	port    int
	url     string
	running bool
}

// NewTTYDServer creates a new ttyd server to run a shell (VHS approach)
func NewTTYDServer() (*TTYDServer, error) {
	// Get random available port
	port, err := getRandomPort()
	if err != nil {
		return nil, fmt.Errorf("failed to get random port: %w", err)
	}

	// Build ttyd command (exactly like VHS tty.go) - launch shell, not binary
	args := []string{
		"--port", fmt.Sprintf("%d", port),
		"--interface", "127.0.0.1",
		"-t", "rendererType=canvas",
		"-t", "disableResizeOverlay=true",
		"-t", "enableSixel=true",
		"-t", "customGlyphs=true",
		"--once",
		"--writable",
		"/bin/zsh",  // Launch shell (VHS approach) instead of binary directly
	}

	cmd := exec.Command("ttyd", args...)

	server := &TTYDServer{
		cmd:  cmd,
		port: port,
		url:  fmt.Sprintf("http://127.0.0.1:%d", port),
	}

	return server, nil
}

// Start launches the ttyd server
func (s *TTYDServer) Start() error {
	if s.running {
		return fmt.Errorf("ttyd server already running")
	}

	// Start ttyd in background
	if err := s.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start ttyd: %w", err)
	}

	s.running = true

	// Wait for server to be ready
	if err := s.waitForServer(10 * time.Second); err != nil {
		s.Stop()
		return fmt.Errorf("ttyd server failed to start: %w", err)
	}

	fmt.Printf("🌐 ttyd server started at %s\n", s.url)
	return nil
}

// Stop terminates the ttyd server
func (s *TTYDServer) Stop() error {
	if !s.running {
		return nil
	}

	s.running = false

	if s.cmd.Process != nil {
		if err := s.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill ttyd process: %w", err)
		}
	}

	// Wait for process to exit
	s.cmd.Wait()
	fmt.Println("🛑 ttyd server stopped")
	return nil
}

// GetURL returns the ttyd server URL
func (s *TTYDServer) GetURL() string {
	return s.url
}

// GetPort returns the ttyd server port
func (s *TTYDServer) GetPort() int {
	return s.port
}

// IsRunning checks if the ttyd server is running
func (s *TTYDServer) IsRunning() bool {
	return s.running
}

// waitForServer waits for ttyd to become available
func (s *TTYDServer) waitForServer(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", s.port), time.Second)
		if err == nil {
			conn.Close()
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	
	return fmt.Errorf("ttyd server did not start within %v", timeout)
}

// getRandomPort returns an available port (like VHS randomPort function)
func getRandomPort() (int, error) {
	listener, err := net.Listen("tcp", ":0")
	if err != nil {
		return 0, err
	}
	defer listener.Close()
	
	addr := listener.Addr().(*net.TCPAddr)
	return addr.Port, nil
}