import sys
import os
import threading

# Import necessary PyQt5 modules
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                             QTextEdit, QInputDialog, QPushButton, QMenu, QAction,
                             QSizePolicy)
# Import necessary QtGui modules, including QContextMenuEvent
from PyQt5.QtGui import (QPainter, QBrush, QColor, QMouseEvent, QRadialGradient,
                         QCursor, QContextMenuEvent)
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QThread, QPoint, QEasingCurve, QPropertyAnimation, QTimer

import math # Needed for distance calculation

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


# --- Gemini API Interaction in a Separate Thread ---
# Handles the API call off the main GUI thread to keep the GUI responsive.

class GeminiThread(QThread):
    # Signals to communicate back to the main GUI thread
    response_signal = pyqtSignal(str) # Emits the successful response text
    error_signal = pyqtSignal(str) # Emits an error message

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt

    def run(self):
        """Runs the Gemini API call in this separate thread."""
        try:
            # --- FOLLOW WORKING EXAMPLE: Instantiate genai.Client ---
            client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"),
            )
            # ----------------------------------------------------------

            # --- FOLLOW WORKING EXAMPLE: Define model name as a string ---
            # Use the model name you had in your working example, or 'gemini-1.5-flash-latest'
            # Let's use 'gemini-1.5-flash-latest' for consistency unless specified otherwise
            model_name = 'gemini-1.5-flash-latest'
            # --------------------------------------------------------------

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=self.prompt),
                    ],
                ),
            ]

            # --- FOLLOW WORKING EXAMPLE: Call generate_content using client.models and model=string ---
            # Note: Your working example uses generate_content_stream.
            # For simplicity with QThread, we'll use generate_content which is non-streaming,
            # but the principle of passing model as a string remains the same.
            # If streaming is desired, the QThread implementation would need adjustment
            # to process chunks as they arrive.
            response = client.models.generate_content(model=model_name, contents=contents)
            # -------------------------------------------------------------------------------------

            # Process the response
            if response and response.text:
                self.response_signal.emit(response.text) # Send response text back to GUI
            else:
                 self.response_signal.emit("Avra returned an empty or unexpected response.")

        except Exception as e:
            self.error_signal.emit(f"Avra API Error: {e}") # Send error message back to GUI


# --- Custom Widget for the Clickable Gradient Circle ---
# This widget draws the gradient circle and handles clicks within it.
# It will be part of the main window.

class GradientCircleWidget(QWidget):
    circle_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._circle_radius = 75 # Desired visual radius
        self.setMinimumSize(self._circle_radius * 2, self._circle_radius * 2)

    def paintEvent(self, event):
        """Draws the gradient circle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Center the circle drawing based on current widget size
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2

        # Create a radial gradient
        gradient = QRadialGradient(center_x, center_y, self._circle_radius,
                                   center_x, center_y)
        gradient.setColorAt(0.0, QColor(150, 200, 255, 200)) # Inner color (semi-transparent blue)
        gradient.setColorAt(0.8, QColor(50, 100, 150, 150)) # Outer color (more opaque blue)
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0)) # Fully transparent at the edge

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen) # No border

        # Draw the ellipse based on the calculated center and radius
        painter.drawEllipse(QPoint(center_x, center_y), self._circle_radius, self._circle_radius)

    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse press events to detect clicks on the circle."""
        if event.button() == Qt.LeftButton:
            # Check if the click was inside the visually drawn circle area
            width = self.width()
            height = self.height()
            center_x = width // 2
            center_y = height // 2
            click_pos = event.pos()

            # Calculate Euclidean distance from the center
            distance_from_center = math.sqrt(
                (click_pos.x() - center_x)**2 + (click_pos.y() - center_y)**2
            )

            if distance_from_center <= self._circle_radius:
                print("Gradient circle clicked!")
                self.circle_clicked.emit() # Emit the signal if clicked inside the circle
            else:
                 # Pass other clicks to the parent for window dragging
                 super().mousePressEvent(event)

        else:
            # Pass non-left clicks (like right-click) to the parent
            super().mousePressEvent(event)


# --- Main Application Window (Borderless, Transparent, Draggable) ---

class AvraScreenBot(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Avra Screen Bot")
        # Make the window frameless, stay on top, and have a tool window style (no taskbar entry)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground) # Enable transparency

        # Layout - Use a VBox, but items will be centered
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0) # Remove margins
        main_layout.setSpacing(0) # Remove spacing

        # Gradient Circle Widget instance
        self.circle_widget = GradientCircleWidget()
        # Connect the circle's clicked signal to the slot that handles prompting and API call
        self.circle_widget.circle_clicked.connect(self.on_circle_clicked)
        main_layout.addWidget(self.circle_widget, alignment=Qt.AlignCenter) # Center the circle widget

        # Response Display (initially hidden or very small)
        self.response_text_edit = QTextEdit()
        self.response_text_edit.setReadOnly(True)
        self.response_text_edit.setMaximumHeight(0) # Start hidden
        self.response_text_edit.setMinimumHeight(0)
        self.response_text_edit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.response_text_edit.setStyleSheet("background-color: rgba(255, 255, 255, 150); border: 1px solid rgba(0, 0, 0, 50);") # Semi-transparent background
        main_layout.addWidget(self.response_text_edit, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

        self._drag_position = None # To store the position when dragging starts
        self.gemini_thread = None # Initialize thread attribute

    # Use QContextMenuEvent directly after importing it
    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create and show a context menu on right-click."""
        context_menu = QMenu(self)
        close_action = QAction("Close Avra", self)
        close_action.triggered.connect(self.close) # Connect to the window's close method
        context_menu.addAction(close_action)

        # Show the menu at the cursor position
        context_menu.exec_(QCursor.pos())

    def on_circle_clicked(self):
        """Slot triggered when the circle is clicked. Opens a prompt dialog."""
        print("Circle clicked! Opening prompt dialog.")
        # Open a text input dialog - it will appear as a standard window
        prompt, ok = QInputDialog.getText(self, "Avra Prompt", "Enter your message for Avra:")

        if ok and prompt: # If the user clicked OK and entered text
            print(f"User entered prompt: '{prompt}'")

            # Show the response text edit area and indicate thinking
            self.response_text_edit.setMaximumHeight(150) # Expand the text area
            self.response_text_edit.setMinimumHeight(50)
            self.response_text_edit.setText("Avra is thinking...")
            QApplication.processEvents() # Process events to update the GUI immediately

            # Start the Gemini API call in a new thread
            self.gemini_thread = GeminiThread(prompt)
            # Connect signals from the thread to update the GUI
            self.gemini_thread.response_signal.connect(self.display_response)
            self.gemini_thread.error_signal.connect(self.display_error)
            self.gemini_thread.finished.connect(self.on_gemini_thread_finished) # Connect finished signal
            self.gemini_thread.start() # Start the thread

    def display_response(self, response_text: str):
        """Slot to update the response text edit with Avra's response."""
        print("Received response from Avra thread.")
        self.response_text_edit.setText(response_text)

    def display_error(self, error_message: str):
        """Slot to update the response text edit with an error message."""
        print(f"Received error from Avra thread: {error_message}")
        self.response_text_edit.setText(f"Error: {error_message}")

    def on_gemini_thread_finished(self):
        """Slot called when the Gemini thread finishes."""
        print("Gemini thread finished.")
        # Clean up the thread reference
        self.gemini_thread = None
        # Optionally, shrink the response box if the user wants
        # self.response_text_edit.setMaximumHeight(0)
        # self.response_text_edit.setMinimumHeight(0)


# --- Main Application Entry Point ---

if __name__ == "__main__":
    # Ensure the API key is set before starting
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set the GEMINI_API_KEY in your .env file or environment variables.")
        sys.exit(1)

    # Create the PyQt application instance
    app = QApplication(sys.argv)

    # Create the main window instance
    main_window = AvraScreenBot()

    # Show the main window
    main_window.show()

    # Start the Qt application event loop
    sys.exit(app.exec_())