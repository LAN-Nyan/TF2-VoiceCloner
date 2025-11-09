#!/usr/bin/env python3
"""
TF2 Voice Line Editor - Desktop Application with Auto-Installer
Standalone GUI app for voice line transcription and AI generation
Build: pyinstaller --onefile --windowed --name "TF2VoiceEditor" tf2_voice_editor.py
"""

import sys
import os
import json
import subprocess
import hashlib
import threading
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QLineEdit, QTextEdit, QListWidget, QListWidgetItem,
    QComboBox, QFileDialog, QProgressBar, QScrollArea, QFrame, QMessageBox,
    QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon


def get_marker_path():
    """Get the path for the installation marker file"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(app_dir, ".tf2voice_installed")


def check_installation():
    """Check if dependencies are already installed"""
    marker_path = get_marker_path()
    
    # First check marker file
    if os.path.exists(marker_path):
        try:
            with open(marker_path, 'r') as f:
                content = f.read().strip()
                if content == "installed":
                    # Also verify critical packages can be imported
                    try:
                        import torch
                        return True
                    except ImportError:
                        # Marker exists but packages missing - delete marker
                        os.remove(marker_path)
                        return False
        except Exception:
            pass
    
    # Check if we can import critical packages
    try:
        import torch
        import f5_tts
        import pydub
        import vosk
        # If we can import them, create the marker
        try:
            with open(marker_path, 'w') as f:
                f.write("installed")
        except Exception:
            pass
        return True
    except ImportError:
        return False


class DependencyInstaller(QDialog):
    """Dialog for installing dependencies on first run"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Installing Dependencies")
        self.setModal(True)
        self.setMinimumSize(600, 350)
        self.installation_complete = False
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title = QLabel("First-Time Setup")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Installing AI packages (torch, F5-TTS, vosk, etc.)\n"
            "This only happens once and may take 5-10 minutes.\n\n"
            "⚠️ Please keep this window open until installation completes!"
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Current package label
        self.package_label = QLabel("Preparing installation...")
        self.package_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.package_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.package_label)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        self.setStyleSheet("""
            QDialog {
                background-color: #1f2937;
            }
            QLabel {
                color: white;
            }
            QProgressBar {
                border: 2px solid #374151;
                border-radius: 5px;
                text-align: center;
                background-color: #374151;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #fb923c;
            }
        """)
    
    def closeEvent(self, event):
        """Prevent closing during installation"""
        if not self.installation_complete:
            reply = QMessageBox.question(
                self, 
                'Installation in Progress',
                'Installation is not complete. Closing now may cause issues.\n\nAre you sure you want to close?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        event.accept()


class InstallWorker(QThread):
    """Worker thread for installing dependencies"""
    progress = pyqtSignal(int, str, str)  # percent, package_name, status
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self):
        super().__init__()
        self.packages = [
            ("torch", "PyTorch (Deep Learning Framework)"),
            ("torchaudio", "Audio Processing for PyTorch"),
            ("f5-tts", "F5-TTS (Voice Cloning AI)"),
            ("soundfile", "Audio File I/O"),
            ("pydub", "Audio Manipulation"),
            ("vosk", "Speech Recognition"),
            ("requests", "HTTP Library")
        ]
        
    def run(self):
        """Install all packages"""
        try:
            total = len(self.packages)
            
            for idx, (package, description) in enumerate(self.packages):
                percent = int((idx / total) * 100)
                self.progress.emit(percent, package, f"Installing {description}...")
                
                # Install package with explicit timeout and error handling
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package, "--no-warn-script-location"],
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minute timeout per package
                    )
                    
                    if result.returncode != 0:
                        error_msg = result.stderr if result.stderr else "Unknown error"
                        self.finished.emit(False, f"Failed to install {package}:\n{error_msg[:500]}")
                        return
                    
                    self.progress.emit(
                        int(((idx + 1) / total) * 100), 
                        package, 
                        f"✓ Installed {description}"
                    )
                    
                except subprocess.TimeoutExpired:
                    self.finished.emit(False, f"Installation of {package} timed out. Please check your internet connection.")
                    return
            
            # Mark as installed BEFORE signaling finish
            if self.mark_installed():
                self.finished.emit(True, "All packages installed successfully! 🎉")
            else:
                self.finished.emit(False, "Installation completed but could not create marker file. You may need to run as administrator.")
            
        except Exception as e:
            self.finished.emit(False, f"Installation error: {str(e)}")
    
    def mark_installed(self):
        """Create marker file to indicate installation is complete"""
        try:
            marker_path = get_marker_path()
            with open(marker_path, "w") as f:
                f.write("installed")
            return True
        except Exception as e:
            print(f"Could not create marker file: {e}")
            return False


def show_installer():
    """Show installer dialog and install dependencies"""
    dialog = DependencyInstaller()
    worker = InstallWorker()
    
    def update_progress(percent, package, status):
        dialog.progress.setValue(percent)
        dialog.package_label.setText(f"Installing: {package}")
        dialog.status_label.setText(status)
    
    def installation_finished(success, message):
        dialog.installation_complete = True
        if success:
            QMessageBox.information(dialog, "Success", message + "\n\nThe application will now start.")
            dialog.accept()
        else:
            QMessageBox.critical(dialog, "Installation Failed", 
                message + "\n\nYou can try:\n"
                "1. Running as administrator\n"
                "2. Checking your internet connection\n"
                "3. Installing manually: pip install torch torchaudio f5-tts soundfile pydub vosk requests")
            dialog.reject()
    
    worker.progress.connect(update_progress)
    worker.finished.connect(installation_finished)
    worker.start()
    
    result = dialog.exec()
    
    # Make sure thread is finished
    worker.wait()
    
    return result == QDialog.DialogCode.Accepted


# Now import the optional packages
try:
    import torch
    from f5_tts.api import F5TTS
    F5_AVAILABLE = True
except:
    F5_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except:
    PYDUB_AVAILABLE = False

try:
    from vosk import Model, KaldiRecognizer
    import wave
    VOSK_AVAILABLE = True
except:
    VOSK_AVAILABLE = False


class TranscriptionWorker(QThread):
    """Worker thread for transcribing audio files"""
    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal(list)  # list of transcribed files
    
    def __init__(self, files):
        super().__init__()
        self.files = files
        self.transcriptions = []
        
    def run(self):
        """Transcribe all audio files"""
        total = len(self.files)
        
        for idx, filepath in enumerate(self.files):
            filename = os.path.basename(filepath)
            self.progress.emit(idx + 1, total, filename)
            
            # Extract character from path
            character = self.detect_character(filepath)
            
            # Simple transcription (placeholder - would use Vosk/Whisper in production)
            transcript = self.transcribe_file(filepath)
            
            self.transcriptions.append({
                'filepath': filepath,
                'filename': filename,
                'character': character,
                'transcript': transcript
            })
        
        self.finished.emit(self.transcriptions)
    
    def detect_character(self, filepath):
        """Detect TF2 character from filepath"""
        characters = ["Scout", "Soldier", "Pyro", "Demoman", 
                     "Heavy", "Engineer", "Medic", "Sniper", "Spy"]
        
        path_lower = filepath.lower()
        for char in characters:
            if char.lower() in path_lower:
                return char
        return "Unknown"
    
    def transcribe_file(self, filepath):
        """Transcribe audio file (placeholder implementation)"""
        # In production, use Vosk or Whisper here
        # For now, return filename as placeholder
        return f"[Audio from {os.path.basename(filepath)}]"


class GenerationWorker(QThread):
    """Worker thread for AI voice generation"""
    finished = pyqtSignal(str)  # output filepath
    error = pyqtSignal(str)  # error message
    
    def __init__(self, text, reference_audio, output_dir):
        super().__init__()
        self.text = text
        self.reference_audio = reference_audio
        self.output_dir = output_dir
        
    def run(self):
        """Generate AI voice"""
        try:
            if not F5_AVAILABLE:
                self.error.emit("F5-TTS not installed")
                return
            
            # Initialize F5-TTS
            tts = F5TTS()
            
            # Generate output filename
            text_hash = hashlib.md5(self.text.encode()).hexdigest()[:8]
            output_file = os.path.join(self.output_dir, f"generated_{text_hash}.wav")
            
            # Generate speech
            tts.infer(
                ref_file=self.reference_audio,
                ref_text="",
                gen_text=self.text,
                file_wave=output_file,
                remove_silence=True
            )
            
            self.finished.emit(output_file)
        except Exception as e:
            self.error.emit(str(e))


class CombineWorker(QThread):
    """Worker thread for combining audio clips"""
    finished = pyqtSignal(str)  # output filepath
    error = pyqtSignal(str)
    
    def __init__(self, clips, output_dir):
        super().__init__()
        self.clips = clips
        self.output_dir = output_dir
        
    def run(self):
        """Combine audio clips"""
        try:
            if not PYDUB_AVAILABLE:
                self.error.emit("pydub not installed")
                return
            
            combined = AudioSegment.empty()
            
            for clip in self.clips:
                audio = AudioSegment.from_file(clip['filepath'])
                combined += audio
            
            output_file = os.path.join(self.output_dir, "combined_output.wav")
            combined.export(output_file, format="wav")
            
            self.finished.emit(output_file)
        except Exception as e:
            self.error.emit(str(e))


class TF2VoiceEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.transcriptions = []
        self.selected_clips = []
        self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("TF2 Voice Line Editor")
        self.setMinimumSize(1200, 800)
        
        # Set dark theme
        self.set_dark_theme()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        self.create_header(main_layout)
        
        # Upload section
        self.create_upload_section(main_layout)
        
        # Main content (two columns)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left column - Search & Browse
        self.create_search_section(content_layout)
        
        # Right column - Editor
        self.create_editor_section(content_layout)
        
        main_layout.addLayout(content_layout)
        
    def set_dark_theme(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #7c2d12, stop:0.5 #991b1b, stop:1 #1f2937);
            }
            QLabel {
                color: white;
            }
            QLineEdit, QTextEdit, QComboBox, QListWidget {
                background-color: #374151;
                color: white;
                border: 1px solid #4b5563;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton {
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QPushButton:disabled {
                background-color: #4b5563;
                color: #9ca3af;
            }
            QFrame {
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 12px;
            }
        """)
        
    def create_header(self, layout):
        """Create header section"""
        header_layout = QVBoxLayout()
        header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Title
        title = QLabel("TF2 Voice Line Editor")
        title.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        title.setStyleSheet("color: #fb923c;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("AI-powered voice line transcription and generation")
        subtitle.setFont(QFont("Arial", 14))
        subtitle.setStyleSheet("color: #9ca3af;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle)
        
        # Scan button
        scan_btn = QPushButton("🔍 Scan Directory for Audio Files")
        scan_btn.setStyleSheet("background-color: #7c3aed; color: white;")
        scan_btn.clicked.connect(self.scan_directory)
        header_layout.addWidget(scan_btn)
        
        layout.addLayout(header_layout)
        
    def create_upload_section(self, layout):
        """Create upload section"""
        upload_frame = QFrame()
        upload_layout = QVBoxLayout(upload_frame)
        upload_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Upload icon and text
        upload_label = QLabel("☁️ Upload TF2 Audio Files")
        upload_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        upload_label.setStyleSheet("color: #fb923c;")
        upload_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(upload_label)
        
        upload_subtext = QLabel("WAV or MP3 - AI will transcribe automatically")
        upload_subtext.setStyleSheet("color: #9ca3af;")
        upload_subtext.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(upload_subtext)
        
        # Upload button
        upload_btn = QPushButton("Select Files")
        upload_btn.setStyleSheet("background-color: #fb923c; color: white;")
        upload_btn.clicked.connect(self.upload_files)
        upload_layout.addWidget(upload_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #374151;
                border-radius: 5px;
                text-align: center;
                background-color: #1f2937;
            }
            QProgressBar::chunk {
                background-color: #fb923c;
            }
        """)
        self.progress_bar.hide()
        upload_layout.addWidget(self.progress_bar)
        
        # File counter
        self.file_counter = QLabel("0 files loaded and transcribed")
        self.file_counter.setStyleSheet("color: #9ca3af;")
        self.file_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(self.file_counter)
        
        layout.addWidget(upload_frame)
        
    def create_search_section(self, layout):
        """Create search and browse section"""
        search_widget = QWidget()
        search_layout = QVBoxLayout(search_widget)
        
        # Header
        header = QLabel("Search Voice Lines")
        header.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header.setStyleSheet("color: #fb923c;")
        search_layout.addWidget(header)
        
        # Search bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search transcriptions (e.g., 'hello world', 'nice shot')...")
        self.search_input.textChanged.connect(self.filter_results)
        search_layout.addWidget(self.search_input)
        
        # Character filter
        self.character_filter = QComboBox()
        self.character_filter.addItems([
            "All Characters", "Scout", "Soldier", "Pyro", "Demoman",
            "Heavy", "Engineer", "Medic", "Sniper", "Spy"
        ])
        self.character_filter.currentTextChanged.connect(self.filter_results)
        search_layout.addWidget(self.character_filter)
        
        # Results list
        self.results_list = QListWidget()
        self.results_list.setStyleSheet("""
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #374151;
            }
            QListWidget::item:hover {
                background-color: #374151;
            }
        """)
        self.results_list.itemClicked.connect(self.add_to_selection)
        search_layout.addWidget(self.results_list)
        
        layout.addWidget(search_widget, 2)
        
    def create_editor_section(self, layout):
        """Create editor panel"""
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)
        
        # Header
        header = QLabel("Editor")
        header.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header.setStyleSheet("color: #fb923c;")
        editor_layout.addWidget(header)
        
        # AI Voice Cloning section
        ai_frame = QFrame()
        ai_layout = QVBoxLayout(ai_frame)
        
        ai_header = QLabel("🎤 AI Voice Cloning")
        ai_header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        ai_layout.addWidget(ai_header)
        
        ai_subtext = QLabel("Uses your uploaded audio as voice reference!")
        ai_subtext.setStyleSheet("color: #9ca3af; font-size: 12px;")
        ai_layout.addWidget(ai_subtext)
        
        # Voice sample dropdown
        voice_label = QLabel("Voice Sample")
        voice_label.setStyleSheet("font-weight: bold;")
        ai_layout.addWidget(voice_label)
        
        self.voice_sample_combo = QComboBox()
        self.voice_sample_combo.addItems(["Auto (match character)", "None (default voice)"])
        ai_layout.addWidget(self.voice_sample_combo)
        
        # Text input
        self.ai_text_input = QTextEdit()
        self.ai_text_input.setPlaceholderText("Type any text to generate TF2 voice...")
        self.ai_text_input.setMaximumHeight(100)
        ai_layout.addWidget(self.ai_text_input)
        
        # Generate button
        self.generate_btn = QPushButton("Generate with AI")
        self.generate_btn.setStyleSheet("background-color: #fb923c; color: white;")
        self.generate_btn.clicked.connect(self.generate_ai_voice)
        ai_layout.addWidget(self.generate_btn)
        
        editor_layout.addWidget(ai_frame)
        
        # Selected clips section
        clips_label = QLabel("Selected Clips (0)")
        clips_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        editor_layout.addWidget(clips_label)
        self.clips_label = clips_label
        
        self.selected_list = QListWidget()
        self.selected_list.setMaximumHeight(200)
        editor_layout.addWidget(self.selected_list)
        
        # Combine button
        self.combine_btn = QPushButton("Combine Clips")
        self.combine_btn.setStyleSheet("background-color: #10b981; color: white;")
        self.combine_btn.setEnabled(False)
        self.combine_btn.clicked.connect(self.combine_clips)
        editor_layout.addWidget(self.combine_btn)
        
        # Result section
        self.result_label = QLabel("Result")
        self.result_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.result_label.hide()
        editor_layout.addWidget(self.result_label)
        
        self.result_file = None
        self.download_btn = QPushButton("📥 Download WAV")
        self.download_btn.setStyleSheet("background-color: #3b82f6; color: white;")
        self.download_btn.clicked.connect(self.download_result)
        self.download_btn.hide()
        editor_layout.addWidget(self.download_btn)
        
        editor_layout.addStretch()
        
        layout.addWidget(editor_widget, 1)
        
    def scan_directory(self):
        """Scan directory for audio files"""
        directory = QFileDialog.getExistingDirectory(self, "Select Audio Directory")
        if directory:
            self.load_from_directory(directory)
            
    def load_from_directory(self, directory):
        """Load audio files from directory"""
        audio_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    audio_files.append(os.path.join(root, file))
        
        if audio_files:
            self.transcribe_files(audio_files)
        else:
            QMessageBox.warning(self, "No Files", "No audio files found in directory")
            
    def upload_files(self):
        """Upload audio files"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "", 
            "Audio Files (*.wav *.mp3 *.ogg *.flac)"
        )
        if files:
            self.transcribe_files(files)
            
    def transcribe_files(self, files):
        """Transcribe audio files"""
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        self.worker = TranscriptionWorker(files)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.transcription_finished)
        self.worker.start()
        
    def update_progress(self, current, total, filename):
        """Update progress bar"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        
    def transcription_finished(self, transcriptions):
        """Handle transcription completion"""
        self.transcriptions.extend(transcriptions)
        self.progress_bar.hide()
        self.file_counter.setText(f"{len(self.transcriptions)} files loaded and transcribed")
        self.update_results_list()
        
        # Update voice sample dropdown
        self.voice_sample_combo.clear()
        self.voice_sample_combo.addItems(["Auto (match character)", "None (default voice)"])
        for t in self.transcriptions:
            self.voice_sample_combo.addItem(f"{t['filename']} ({t['character']})")
            
    def update_results_list(self):
        """Update the results list"""
        self.results_list.clear()
        
        search_text = self.search_input.text().lower()
        character_filter = self.character_filter.currentText()
        
        for t in self.transcriptions:
            # Apply filters
            if search_text and search_text not in t['transcript'].lower():
                continue
            if character_filter != "All Characters" and t['character'] != character_filter:
                continue
                
            item = QListWidgetItem()
            item_text = f"{t['filename']}\n\"{t['transcript']}\"\n{t['character']} ➕"
            item.setText(item_text)
            item.setData(Qt.ItemDataRole.UserRole, t)
            self.results_list.addItem(item)
            
    def filter_results(self):
        """Filter results based on search and character"""
        self.update_results_list()
        
    def add_to_selection(self, item):
        """Add clip to selection"""
        clip = item.data(Qt.ItemDataRole.UserRole)
        if clip not in self.selected_clips:
            self.selected_clips.append(clip)
            self.update_selected_list()
            
    def update_selected_list(self):
        """Update selected clips list"""
        self.selected_list.clear()
        self.clips_label.setText(f"Selected Clips ({len(self.selected_clips)})")
        
        for idx, clip in enumerate(self.selected_clips, 1):
            item = QListWidgetItem()
            item.setText(f"{idx}. {clip['transcript']} - {clip['character']} ❌")
            item.setData(Qt.ItemDataRole.UserRole, idx - 1)
            self.selected_list.addItem(item)
        
        self.combine_btn.setEnabled(len(self.selected_clips) > 0)
        
    def generate_ai_voice(self):
        """Generate AI voice"""
        text = self.ai_text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter text to generate")
            return
            
        if not F5_AVAILABLE:
            QMessageBox.critical(self, "Error", "F5-TTS not installed. Install with: pip install f5-tts")
            return
            
        # Get reference audio
        voice_idx = self.voice_sample_combo.currentIndex()
        if voice_idx <= 1 or not self.transcriptions:
            QMessageBox.warning(self, "No Reference", "Please select a voice sample")
            return
            
        reference = self.transcriptions[voice_idx - 2]['filepath']
        
        self.generate_btn.setText("Generating...")
        self.generate_btn.setEnabled(False)
        
        self.gen_worker = GenerationWorker(text, reference, self.output_dir)
        self.gen_worker.finished.connect(self.generation_finished)
        self.gen_worker.error.connect(self.generation_error)
        self.gen_worker.start()
        
    def generation_finished(self, output_file):
        """Handle generation completion"""
        self.generate_btn.setText("Generate with AI")
        self.generate_btn.setEnabled(True)
        self.show_result(output_file)
        
    def generation_error(self, error):
        """Handle generation error"""
        self.generate_btn.setText("Generate with AI")
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Generation Error", error)
        
    def combine_clips(self):
        """Combine selected clips"""
        if not PYDUB_AVAILABLE:
            QMessageBox.critical(self, "Error", "pydub not installed. Install with: pip install pydub")
            return
            
        self.combine_btn.setText("Processing...")
        self.combine_btn.setEnabled(False)
        
        self.combine_worker = CombineWorker(self.selected_clips, self.output_dir)
        self.combine_worker.finished.connect(self.combine_finished)
        self.combine_worker.error.connect(self.combine_error)
        self.combine_worker.start()
        
    def combine_finished(self, output_file):
        """Handle combine completion"""
        self.combine_btn.setText("Combine Clips")
        self.combine_btn.setEnabled(True)
        self.show_result(output_file)
        
    def combine_error(self, error):
        """Handle combine error"""
        self.combine_btn.setText("Combine Clips")
        self.combine_btn.setEnabled(True)
        QMessageBox.critical(self, "Combine Error", error)
        
    def show_result(self, filepath):
        """Show result file"""
        self.result_file = filepath
        self.result_label.show()
        self.download_btn.show()
        QMessageBox.information(self, "Success", f"Audio saved to:\n{filepath}")
        
    def download_result(self):
        """Download result file"""
        if self.result_file:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Audio File", "output.wav", "WAV Files (*.wav)"
            )
            if save_path:
                import shutil
                shutil.copy(self.result_file, save_path)
                QMessageBox.information(self, "Saved", f"File saved to:\n{save_path}")


def main():
    # Create QApplication first
    app = QApplication(sys.argv)
    app.setApplicationName("TF2 Voice Line Editor")
    
    # Check if dependencies need to be installed (only once per app instance)
    if not check_installation():
        print("Dependencies not found. Starting installer...")
        if not show_installer():
            QMessageBox.critical(
                None, 
                "Installation Failed", 
                "Could not install required packages.\n\n"
                "Please install manually:\n"
                "pip install torch torchaudio f5-tts soundfile pydub vosk requests\n\n"
                "Or run as administrator and try again."
            )
            sys.exit(1)
        
        # Verify installation succeeded
        if not check_installation():
            QMessageBox.critical(
                None,
                "Verification Failed",
                "Installation completed but packages could not be verified.\n\n"
                "Please restart the application."
            )
            sys.exit(1)
    
    # Only create main window after installation check
    window = TF2VoiceEditor()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
