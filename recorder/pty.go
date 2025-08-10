package recorder

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"time"

	"github.com/creack/pty"
)

// PTYSession manages a pseudo-terminal session running the karaoke TUI
type PTYSession struct {
	pty    *os.File
	tty    *os.File
	cmd    *exec.Cmd
	output chan []byte
	done   chan bool
}

// NewPTYSession creates a new PTY session to run the karaoke TUI
func NewPTYSession(binaryPath string) (*PTYSession, error) {
	// Create command to run karaoke TUI binary with audio enabled
	cmd := exec.Command(binaryPath, "-audio")

	// Start command with PTY
	ptmx, err := pty.Start(cmd)
	if err != nil {
		return nil, fmt.Errorf("failed to start PTY: %w", err)
	}

	session := &PTYSession{
		pty:    ptmx,
		cmd:    cmd,
		output: make(chan []byte, 1000),
		done:   make(chan bool),
	}

	// Start reading PTY output in background
	go session.readOutput()

	return session, nil
}

// readOutput continuously reads from PTY and sends to output channel
func (s *PTYSession) readOutput() {
	defer close(s.output)

	reader := bufio.NewReader(s.pty)
	buffer := make([]byte, 4096)

	for {
		select {
		case <-s.done:
			return
		default:
			n, err := reader.Read(buffer)
			if err != nil {
				if err == io.EOF {
					return
				}
				continue
			}
			if n > 0 {
				// Send copy of buffer data to channel
				data := make([]byte, n)
				copy(data, buffer[:n])
				select {
				case s.output <- data:
				case <-s.done:
					return
				}
			}
		}
	}
}

// GetOutput returns the output channel for reading terminal data
func (s *PTYSession) GetOutput() <-chan []byte {
	return s.output
}

// SetSize sets the PTY window size to match expected terminal dimensions
func (s *PTYSession) SetSize(cols, rows uint16) error {
	return pty.Setsize(s.pty, &pty.Winsize{
		Cols: cols,
		Rows: rows,
	})
}

// Close stops the PTY session and cleans up resources
func (s *PTYSession) Close() error {
	close(s.done)

	// Give the process time to finish gracefully
	time.Sleep(100 * time.Millisecond)

	if s.cmd.Process != nil {
		s.cmd.Process.Kill()
	}

	if s.pty != nil {
		s.pty.Close()
	}

	return s.cmd.Wait()
}

// IsRunning checks if the PTY process is still active
func (s *PTYSession) IsRunning() bool {
	if s.cmd.Process == nil {
		return false
	}

	// Check if process is still running
	err := s.cmd.Process.Signal(nil)
	return err == nil
}
