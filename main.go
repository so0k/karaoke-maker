package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/faiface/beep/mp3"
	"github.com/faiface/beep/speaker"
)

type Word struct {
	W string  `json:"w"`
	T float64 `json:"t"`
}

type Line struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Words []Word  `json:"words"`
}

type Block struct {
	Type  string  `json:"type"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Lines []Line  `json:"lines"`
}

type TimingData struct {
	Blocks []Block `json:"blocks"`
}

type model struct {
	timingData         TimingData
	currentWordIdx     int
	currentLineIdx     int
	currentBlockIdx    int
	currentBlockType   string
	displayedText      string
	terminalLog        []string
	width              int
	height             int
	startTime          time.Time
	timescale          float64
	chunksStarted      bool
	chunks             []string
	currentChunk       int
	chunkLines         []string
	chunkLineIndex     int
	allWords           []Word
	lyricsViewport     viewport.Model
	terminalViewport   viewport.Model
	chunkPauseUntil    time.Time
	cursorVisible      bool
	cursorCounter      int
	chunkCycleComplete bool
	audioEnabled       bool
	audioStarted       bool
}

type lyricsTickMsg time.Time
type terminalTickMsg time.Time

func lyricsTickCmd() tea.Cmd {
	return tea.Tick(time.Millisecond*50, func(t time.Time) tea.Msg {
		return lyricsTickMsg(t)
	})
}

func terminalTickCmd() tea.Cmd {
	return tea.Tick(time.Millisecond*166, func(t time.Time) tea.Msg {
		return terminalTickMsg(t)
	})
}

// lyricsHeaderLineCount returns how many lines are reserved above the lyrics
// viewport for headers, depending on the current phase.
// - During countdown (before first word), we show title + blank (2 lines)
// - During runtime, we show title + blank + runtime line + blank (4 lines)
func (m model) lyricsHeaderLineCount(elapsed float64) int {
	if m.currentWordIdx == 0 && elapsed < 9.84 {
		return 2
	}
	return 4
}

func (m model) getCurrentBlock(elapsed float64) (int, string) {
	for i, block := range m.timingData.Blocks {
		if elapsed >= block.Start && elapsed <= block.End {
			return i, block.Type
		}
	}
	return m.currentBlockIdx, m.currentBlockType
}

func (m model) getBlockTypeAtTime(timeStamp float64) string {
	for _, block := range m.timingData.Blocks {
		if timeStamp >= block.Start && timeStamp <= block.End {
			return block.Type
		}
	}
	return "verse" // Default to verse if time is before any block
}

func (m model) getAbsoluteLineIndex(wordIdx int) int {
	if wordIdx >= len(m.allWords) {
		return -1
	}

	wordCount := 0
	absoluteLineIdx := 0

	// Find which absolute line this word belongs to
	for _, block := range m.timingData.Blocks {
		for _, line := range block.Lines {
			for range line.Words {
				if wordCount == wordIdx {
					return absoluteLineIdx
				}
				wordCount++
			}
			absoluteLineIdx++
		}
	}
	return -1
}

func (m model) getWordBlockIndex(wordIdx int) int {
	if wordIdx >= len(m.allWords) {
		return -1
	}

	wordCount := 0

	// Find which block this word belongs to
	for blockIdx, block := range m.timingData.Blocks {
		for _, line := range block.Lines {
			for range line.Words {
				if wordCount == wordIdx {
					return blockIdx
				}
				wordCount++
			}
		}
	}
	return -1
}

func startAudio(timescale float64) error {
	file, err := os.Open("audio/from_boilerplate_to_flow.mp3")
	if err != nil {
		return err
	}

	streamer, format, err := mp3.Decode(file)
	if err != nil {
		return err
	}

	speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10))

	// Play the audio in background
	// Note: when timescale != 1.0, audio isn't slowed to match.
	// For capture workflows, prefer running without -audio when timescale != 1.
	if timescale != 1.0 {
		fmt.Fprintf(os.Stderr, "[warn] audio playback not scaled; timescale=%.2f\n", timescale)
	}
	speaker.Play(streamer)

	return nil
}

// Utilities for subtle 80s-style gradients per block
// We compute a gradient color given progress through a block.

// hexToRGB converts #RRGGBB to integer RGB.
func hexToRGB(hex string) (int, int, int) {
	if strings.HasPrefix(hex, "#") {
		hex = hex[1:]
	}
	if len(hex) != 6 {
		return 255, 255, 255
	}
	var r, g, b int
	fmt.Sscanf(hex, "%02x%02x%02x", &r, &g, &b)
	return r, g, b
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func lerp(a, b int, t float64) int {
	return int(float64(a) + (float64(b)-float64(a))*t)
}

func rgbToHex(r, g, b int) string {
	return fmt.Sprintf("#%02x%02x%02x", r, g, b)
}

// gradientColorAt returns an interpolated color from a list of hex stops.
func gradientColorAt(stops []string, t float64) string {
	n := len(stops)
	if n == 0 {
		return "#a6accd"
	}
	if n == 1 {
		return stops[0]
	}
	t = clamp01(t)
	// Determine segment
	seg := int(t * float64(n-1))
	if seg >= n-1 {
		seg = n - 2
	}
	localT := (t*float64(n-1) - float64(seg))
	r1, g1, b1 := hexToRGB(stops[seg])
	r2, g2, b2 := hexToRGB(stops[seg+1])
	r := lerp(r1, r2, localT)
	g := lerp(g1, g2, localT)
	b := lerp(b1, b2, localT)
	return rgbToHex(r, g, b)
}

// getBlockGradientColor picks a subtle neon-80s palette and returns
// a hex color at the given progress [0..1] within the block.
func getBlockGradientColor(blockType string, progress float64) lipgloss.Color {
	progress = clamp01(progress)

	// Palettes: chosen for cool, subtle 80s terminal vibes
	// verse: teal/cyan
	verse := []string{"#6fffe9", "#34e5ff", "#00c2ff"}
	// chorus: lavender/magenta (muted)
	chorus := []string{"#ad8cff", "#c774e8", "#ff6ad5"}
	// bridge: icy blue to periwinkle
	bridge := []string{"#7ae2ff", "#7c83fd", "#a5a1ff"}
	// outro: soft pink/lilac
	outro := []string{"#ff99c8", "#fec8d8", "#cdb4db"}

	var stops []string
	switch blockType {
	case "verse":
		stops = verse
	case "chorus":
		stops = chorus
	case "bridge":
		stops = bridge
	case "outro":
		stops = outro
	default:
		// fallback cool blue gradient
		stops = []string{"#6ec1ff", "#7aa8ff", "#8fb8ff"}
	}
	return lipgloss.Color(gradientColorAt(stops, progress))
}

func stylizeTerminalContent(content string) string {
	// Multi-line aware styling for <error> and <bliss> tags
	// Use DOTALL mode (?s) so "." matches newlines.
	errorRegex := regexp.MustCompile(`(?s)<error>(.*?)</error>`) // red-ish
	content = errorRegex.ReplaceAllStringFunc(content, func(match string) string {
		text := strings.TrimPrefix(strings.TrimSuffix(match, "</error>"), "<error>")
		// #ff8aa1
		return lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Render(text)
	})

	blissRegex := regexp.MustCompile(`(?s)<bliss>(.*?)</bliss>`) // green-ish
	content = blissRegex.ReplaceAllStringFunc(content, func(match string) string {
		text := strings.TrimPrefix(strings.TrimSuffix(match, "</bliss>"), "<bliss>")
		// #8af5c9
		return lipgloss.NewStyle().Foreground(lipgloss.Color("10")).Render(text)
	})

	// Terraform-style change indicators at line start (ignoring leading spaces)
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		// Find first non-space character
		idx := -1
		for j, r := range line {
			if r != ' ' && r != '\t' {
				idx = j
				break
			}
		}
		if idx == -1 {
			continue
		}
		trimmed := strings.TrimLeft(line, " \t")
		first := line[idx]
		var color string
		switch first {
		case '+':
			color = "#8af5c9" // add: pastel green
		case '-':
			// Only highlight true destroy lines, not provider downloads, etc.
			// We require the trimmed line to start with "- resource" specifically.
			if strings.HasPrefix(trimmed, "- resource") {
				color = "#ff8aa1" // destroy: pastel red
			} else {
				continue
			}
		case '~':
			color = "#ffc27d" // modify: pastel orange
		default:
			continue
		}
		lines[i] = lipgloss.NewStyle().Foreground(lipgloss.Color(color)).Render(line)
	}
	return strings.Join(lines, "\n")
}

func loadTimingData() (TimingData, error) {
	data, err := os.ReadFile("audio/from_boilerplate_to_flow_timings.json")
	if err != nil {
		return TimingData{}, err
	}

	var timingData TimingData
	err = json.Unmarshal(data, &timingData)
	return timingData, err
}

func loadChunks() ([]string, error) {
	var chunks []string
	err := filepath.WalkDir("chunks", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if strings.HasSuffix(path, ".txt") {
			chunks = append(chunks, path)
		}
		return nil
	})
	sort.Strings(chunks)
	return chunks, err
}

func initialModel(audioEnabled bool, timescale float64) model {
	timingData, err := loadTimingData()
	if err != nil {
		panic(err)
	}

	chunks, err := loadChunks()
	if err != nil {
		panic(err)
	}

	// Flatten all words with their timings
	var allWords []Word
	for _, block := range timingData.Blocks {
		for _, line := range block.Lines {
			allWords = append(allWords, line.Words...)
		}
	}

	// Initialize viewports with reasonable defaults
	lyricsViewport := viewport.New(70, 30)
	lyricsViewport.SetContent("♪ Instructions\n\nSystem initializing...")

	terminalViewport := viewport.New(50, 30)
	terminalViewport.SetContent("System Output - Awaiting instruction stream...")

	return model{
		timingData:       timingData,
		chunks:           chunks,
		startTime:        time.Now(),
		timescale:        timescale,
		allWords:         allWords,
		lyricsViewport:   lyricsViewport,
		terminalViewport: terminalViewport,
		chunkPauseUntil:  time.Time{},
		audioEnabled:     audioEnabled,
		audioStarted:     false,
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(lyricsTickCmd(), terminalTickCmd())
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		// Update both viewports to match content areas
		usableWidth := m.width - 4 // Account for both pane borders
		lyricsRatio := 0.55
		lyricsWidth := int(float64(usableWidth) * lyricsRatio)
		terminalWidth := usableWidth - lyricsWidth

		// Calculate content dimensions accounting for borders and padding
		lyricsContentWidth := lyricsWidth - 4 // Account for border (2) + padding (2)
		terminalContentWidth := terminalWidth - 4

		// Reserve vertical space for headers rendered outside viewports
		// Also reserve a top margin so the top border isn't flush with screen top
		topMargin := 1
		elapsed := time.Since(m.startTime).Seconds() / m.timescale
		lyricsHeaderLines := m.lyricsHeaderLineCount(elapsed)
		terminalHeaderLines := 2

		lyricsContentHeight := m.height - topMargin - 4 - lyricsHeaderLines
		terminalContentHeight := m.height - topMargin - 4 - terminalHeaderLines
		if lyricsContentHeight < 1 {
			lyricsContentHeight = 1
		}
		if terminalContentHeight < 1 {
			terminalContentHeight = 1
		}

		m.lyricsViewport.Width = lyricsContentWidth
		m.lyricsViewport.Height = lyricsContentHeight
		m.terminalViewport.Width = terminalContentWidth
		m.terminalViewport.Height = terminalContentHeight

		// Update both viewports with the message
		var cmd1, cmd2 tea.Cmd
		m.lyricsViewport, cmd1 = m.lyricsViewport.Update(msg)
		m.terminalViewport, cmd2 = m.terminalViewport.Update(msg)
		return m, tea.Batch(cmd1, cmd2)

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			return m, tea.Quit
		default:
			// Forward key events to both viewports for scrolling
			var cmd1, cmd2 tea.Cmd
			m.lyricsViewport, cmd1 = m.lyricsViewport.Update(msg)
			m.terminalViewport, cmd2 = m.terminalViewport.Update(msg)
			return m, tea.Batch(cmd1, cmd2)
		}

	case lyricsTickMsg:
		elapsed := time.Since(m.startTime).Seconds() / m.timescale

		// Update current block tracking
		blockIdx, blockType := m.getCurrentBlock(elapsed)
		previousBlockType := m.currentBlockType
		m.currentBlockIdx = blockIdx
		m.currentBlockType = blockType

		// Restart chunk cycling when entering a verse block
		if m.chunkCycleComplete && blockType == "verse" && previousBlockType != "verse" {
			m.chunkCycleComplete = false
		}

		// Start audio immediately when countdown begins if audio enabled
		if m.audioEnabled && !m.audioStarted && elapsed >= 0 {
			go func() {
				if err := startAudio(m.timescale); err != nil {
					// Audio failed to start, continue without it
				}
			}()
			m.audioStarted = true
		}

		// Update cursor blinking (toggle every 10 ticks ≈ ~500ms)
		m.cursorCounter++
		if m.cursorCounter >= 10 {
			m.cursorVisible = !m.cursorVisible
			m.cursorCounter = 0
		}

		// Check if we should display the next word
		newWordAdded := false
		if m.currentWordIdx < len(m.allWords) {
			word := m.allWords[m.currentWordIdx]
			if elapsed >= (word.T - 0.5) {
				// Check if we're starting a new line using absolute line positions
				currentLineIdx := m.getAbsoluteLineIndex(m.currentWordIdx)
				previousLineIdx := -1
				currentBlockIdx := -1
				previousBlockIdx := -1

				if m.currentWordIdx > 0 {
					previousLineIdx = m.getAbsoluteLineIndex(m.currentWordIdx - 1)
					previousBlockIdx = m.getWordBlockIndex(m.currentWordIdx - 1)
				}
				if m.currentWordIdx < len(m.allWords) {
					currentBlockIdx = m.getWordBlockIndex(m.currentWordIdx)
				}

				// Add newlines based on line/block transitions
				if currentLineIdx != -1 && previousLineIdx != -1 && currentLineIdx != previousLineIdx && m.currentWordIdx > 0 {
					// Check if this is a block transition (add double newline)
					if currentBlockIdx != -1 && previousBlockIdx != -1 && currentBlockIdx != previousBlockIdx {
						m.displayedText += "\n\n"
					} else {
						// Regular line transition within same block (single newline)
						m.displayedText += "\n"
					}
				} else if len(m.displayedText) > 0 && !strings.HasSuffix(m.displayedText, "\n") {
					// Add space between words on the same line
					m.displayedText += " "
				}

				// Add the word to displayed text with a subtle gradient color
				// based on the word's timing progress within its block
				wordBlockType := m.getBlockTypeAtTime(word.T)
				blockIdx := m.getWordBlockIndex(m.currentWordIdx)
				progress := 0.0
				if blockIdx >= 0 && blockIdx < len(m.timingData.Blocks) {
					blk := m.timingData.Blocks[blockIdx]
					denom := (blk.End - blk.Start)
					if denom > 0 {
						progress = (word.T - blk.Start) / denom
						if progress < 0 {
							progress = 0
						} else if progress > 1 {
							progress = 1
						}
					}
				}
				wordColor := getBlockGradientColor(wordBlockType, progress)
				coloredWord := lipgloss.NewStyle().Foreground(wordColor).Render(word.W)

				m.displayedText += coloredWord
				m.currentWordIdx++
				newWordAdded = true

				// Start chunks after first word
				if !m.chunksStarted && m.currentWordIdx == 1 {
					m.chunksStarted = true
				}
			}
		}

		// Build cursor matching last printed word's color while waiting for next
		cursor := ""
		if m.cursorVisible {
			if m.currentWordIdx < len(m.allWords) {
				nextDisplay := m.allWords[m.currentWordIdx].T - 0.5
				if elapsed < nextDisplay {
					var curColor lipgloss.Color
					if m.currentWordIdx > 0 {
						prevWord := m.allWords[m.currentWordIdx-1]
						bt := m.getBlockTypeAtTime(prevWord.T)
						blkIdx := m.getWordBlockIndex(m.currentWordIdx - 1)
						prog := 0.0
						if blkIdx >= 0 && blkIdx < len(m.timingData.Blocks) {
							blk := m.timingData.Blocks[blkIdx]
							denom := (blk.End - blk.Start)
							if denom > 0 {
								prog = (prevWord.T - blk.Start) / denom
								if prog < 0 {
									prog = 0
								} else if prog > 1 {
									prog = 1
								}
							}
						}
						curColor = getBlockGradientColor(bt, prog)
					} else if len(m.allWords) > 0 {
						nextWord := m.allWords[0]
						bt := m.getBlockTypeAtTime(nextWord.T)
						blkIdx := m.getWordBlockIndex(0)
						prog := 0.0
						if blkIdx >= 0 && blkIdx < len(m.timingData.Blocks) {
							blk := m.timingData.Blocks[blkIdx]
							denom := (blk.End - blk.Start)
							if denom > 0 {
								prog = (nextWord.T - blk.Start) / denom
								if prog < 0 {
									prog = 0
								} else if prog > 1 {
									prog = 1
								}
							}
						}
						curColor = getBlockGradientColor(bt, prog)
					}
					cursor = lipgloss.NewStyle().Bold(true).Foreground(curColor).Render("_")
				}
			}
		}

		// Compose lyrics content inside the viewport only (header is outside)
		var lyricsBody string
		if m.currentWordIdx == 0 && elapsed < 9.84 {
			countdown := 9.84 - elapsed
			lyricsBody = fmt.Sprintf("System initializing... T-%.1f seconds\n\nBoot sequence: %.2fs\nInstruction set: %d entries loaded%s", countdown, elapsed, len(m.allWords), cursor)
		} else {
			lyricsBody = m.displayedText + cursor
		}
		m.lyricsViewport.SetContent(lyricsBody)
		if newWordAdded {
			m.lyricsViewport.GotoBottom()
		}

		// Dynamically resize lyrics viewport height depending on the header size
		topMargin := 1
		lyricsHeaderLines := m.lyricsHeaderLineCount(elapsed)
		lyricsContentHeight := m.height - topMargin - 4 - lyricsHeaderLines
		if lyricsContentHeight < 1 {
			lyricsContentHeight = 1
		}
		if m.lyricsViewport.Height != lyricsContentHeight {
			m.lyricsViewport.Height = lyricsContentHeight
		}

		return m, lyricsTickCmd()

	case terminalTickMsg:
		// Check if we're in a pause between chunks
		if !m.chunkPauseUntil.IsZero() && time.Now().Before(m.chunkPauseUntil) {
			return m, terminalTickCmd()
		}

		// Clear terminal when pause ends (before next chunk)
		if !m.chunkPauseUntil.IsZero() {
			m.terminalLog = nil
			m.terminalViewport.SetContent("")
			m.chunkPauseUntil = time.Time{} // Clear pause
		}

		if m.chunksStarted && (m.currentChunk < len(m.chunks) || !m.chunkCycleComplete) {
			// Load chunk if not loaded
			if len(m.chunkLines) == 0 && m.currentChunk < len(m.chunks) {
				data, err := os.ReadFile(m.chunks[m.currentChunk])
				if err == nil {
					m.chunkLines = strings.Split(string(data), "\n")
					m.chunkLineIndex = 0
				}
			}

			// Add next line to terminal log
			if m.chunkLineIndex < len(m.chunkLines) {
				line := strings.TrimSpace(m.chunkLines[m.chunkLineIndex])
				if line != "" {
					m.terminalLog = append(m.terminalLog, line)
					// Update viewport content with styling
					content := strings.Join(m.terminalLog, "\n")
					styledContent := stylizeTerminalContent(content)
					m.terminalViewport.SetContent(styledContent)
					m.terminalViewport.GotoBottom()
				}
				m.chunkLineIndex++
			} else {
				// Move to next chunk with 3s pause
				m.currentChunk++
				if m.currentChunk >= len(m.chunks) {
					// Reset to first chunk - cycle continuously
					m.currentChunk = 0
					m.chunkCycleComplete = true
					// Only restart cycling immediately if we're in a verse block
					if m.currentBlockType != "verse" {
						// Wait for next verse to restart
						m.chunkCycleComplete = true
					} else {
						m.chunkCycleComplete = false
					}
				}
				m.chunkLines = nil
				m.chunkLineIndex = 0
				m.chunkPauseUntil = time.Now().Add(3 * time.Second)
			}
		} else if !m.chunksStarted {
			// Show waiting message
			m.terminalViewport.SetContent("Awaiting instruction stream activation...")
		}

		return m, terminalTickCmd()
	}

	return m, nil
}

func (m model) View() string {
	if m.width == 0 || m.height == 0 {
		return "Loading..."
	}

	// Calculate pane dimensions for 2-pane layout accounting for borders
	// Each pane has 2-char border (left+right), total 4 chars for both panes
	usableWidth := m.width - 4
	lyricsRatio := 0.55
	lyricsWidth := int(float64(usableWidth) * lyricsRatio)
	terminalWidth := usableWidth - lyricsWidth

	// Styles (a global top spacer line is added above panes in the final render)
	lyricsStyle := lipgloss.NewStyle().
		Width(lyricsWidth).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#cdb4db")).
		Padding(1, 1)

	terminalStyle := lipgloss.NewStyle().
		Width(terminalWidth).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#a5a1ff")).
		Padding(1, 1)

    // Compute runtime for headers (scaled)
    elapsed := time.Since(m.startTime).Seconds() / m.timescale

	// Lyrics header and viewport (reserve space for header outside viewport)
	var lyricsHeader string
	if m.currentWordIdx == 0 && elapsed < 9.84 {
		lyricsHeader = "♪ Instructions\n\n"
	} else {
		lyricsHeader = fmt.Sprintf("♪ Instructions\n\nRuntime: %.2fs | Instruction %d/%d | Block: %s\n\n", elapsed, m.currentWordIdx, len(m.allWords), m.currentBlockType)
	}
	lyricsPane := lyricsStyle.Render(lyricsHeader + m.lyricsViewport.View())

	// Terminal log pane using viewport
	var terminalHeader string
	if m.chunksStarted {
		terminalHeader = fmt.Sprintf("System Output (Module %d/%d)\n\n", m.currentChunk+1, len(m.chunks))
	} else {
		terminalHeader = "System Output (Awaiting instruction stream)\n\n"
	}
	terminalPane := terminalStyle.Render(terminalHeader + m.terminalViewport.View())

	// Combine panes horizontally with a global top spacer so borders are visible
	return "\n" + lipgloss.JoinHorizontal(lipgloss.Top, lyricsPane, terminalPane)
}

func main() {
    audioFlag := flag.Bool("audio", false, "Enable MP3 audio playback")
    timescaleFlag := flag.Float64("timescale", 1.0, "Slow down or speed up timing (e.g., 3.0 = 3x slower)")
    flag.Parse()

    if *timescaleFlag <= 0 {
        fmt.Println("Invalid -timescale; must be > 0")
        os.Exit(1)
    }

    p := tea.NewProgram(
        initialModel(*audioFlag, *timescaleFlag),
        tea.WithAltScreen(),
        tea.WithMouseCellMotion(),
    )

    if _, err := p.Run(); err != nil {
        fmt.Printf("Error: %v", err)
        os.Exit(1)
    }
}
