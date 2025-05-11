import sys
import os
import threading
import math 

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,QTextEdit, QInputDialog, QMenu, QAction,QSizePolicy, QShortcut, QPushButton, QFrame)
from PyQt5.QtGui import (QPainter, QBrush, QColor, QMouseEvent, QRadialGradient,QCursor, QContextMenuEvent, QFont, QKeySequence)
from PyQt5.QtCore import (Qt, QRect, pyqtSignal, QThread, QPoint, QEasingCurve,QPropertyAnimation, QTimer, QEvent, QObject, QSize)
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ----------------------------------------------------------

load_dotenv()

# --- Gemini API Interaction in a Separate Thread ---
# Handles the API call off the main GUI thread to keep the GUI responsive.

class GeminiChatThread(QThread):
    # Signals to communicate back to the main GUI thread
    response_chunk_signal = pyqtSignal(str)
    response_finished_signal = pyqtSignal() # Signal emitted when stream is finished
    error_signal = pyqtSignal(str) # Emits an error message

    def __init__(self, chat_session, prompt: str):
        super().__init__()
        self.chat_session = chat_session
        self.prompt = prompt
        self._is_streaming = False
        print("GeminiChatThread initialized.") # Log for thread creation

    def run(self):
        """Runs the Gemini API call in this separate thread."""
        print("GeminiChatThread running...") # Log for thread start
        try:
            self._is_streaming = True
            # Use send_message_stream for multi-turn conversation
            response = self.chat_session.send_message_stream(self.prompt)

            # Process streaming response
            for chunk in response:
                if chunk.text:
                    # Emit each chunk of the response as it arrives
                    self.response_chunk_signal.emit(chunk.text)
                    # Process events periodically to update the GUI - Might be heavy, reconsider if needed
                    # QApplication.processEvents() # Keep this commented unless absolutely necessary for responsiveness during stream

            self._is_streaming = False
            print("GeminiChatThread received all chunks.")
            self.response_finished_signal.emit() # Signal that streaming is complete

        except Exception as e:
            self._is_streaming = False
            print(f"GeminiChatThread caught API Error: {e}") # Log error
            self.error_signal.emit(f"Avra API Error: {e}")

        print("GeminiChatThread finished.") # Log thread finish

    def is_streaming(self):
        return self._is_streaming


# --- Event Filter to capture Enter key in the input text box ---
class InputKeyFilter(QObject):
    """Filters key press events for the input text edit."""
    enter_pressed = pyqtSignal()

    def eventFilter(self, obj, event):
        """
        Filters events for the watched object.
        Captures the Return/Enter key press (without Shift or Ctrl).
        Allows Shift+Enter for new lines.
        """
        if event.type() == QEvent.KeyPress:
            # Check for Return or Enter key
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                # Check if Shift or Control modifiers are NOT pressed
                if not (event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier)):
                    print("Enter key pressed in input.") # Debugging Enter key
                    self.enter_pressed.emit() # Emit signal
                    return True # Consume the event
        # Pass other events on to the watched object's standard event handlers
        return super().eventFilter(obj, event)


# --- Custom Widget for the Clickable Gradient Circle ---
class GradientCircleWidget(QWidget):
    circle_double_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._circle_radius = 75 # Desired visual radius
        self.setMinimumSize(self._circle_radius * 2, self._circle_radius * 2)
        self.setAttribute(Qt.WA_NoMousePropagation, False)
        self.setMouseTracking(True)

    def paintEvent(self, event):
        """Draws the gradient circle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2

        gradient = QRadialGradient(center_x, center_y, self._circle_radius,
                                   center_x, center_y)
        gradient.setColorAt(0.0, QColor(150, 200, 255, 200)) # Inner color (semi-transparent blue)
        gradient.setColorAt(0.8, QColor(50, 100, 150, 150)) # Outer color (more opaque blue)
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0)) # Fully transparent at the edge

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen) # No border

        painter.drawEllipse(QPoint(center_x, center_y), self._circle_radius, self._circle_radius)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle mouse double-click events within the circle widget."""
        if event.button() == Qt.LeftButton and self._is_click_on_circle(event.pos()):
             print("CircleWidget: Double-click on circle received.")
             self.circle_double_clicked.emit() # Emit signal
             event.accept() # Accept the event as it's handled here
        else:
             event.ignore() # Ignore the event if not handled specifically

    def _is_click_on_circle(self, pos: QPoint) -> bool:
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        distance_from_center = math.sqrt(
            (pos.x() - center_x)**2 + (pos.y() - center_y)**2
        )
        return distance_from_center <= self._circle_radius


# --- Separate Chat Window ---

class AvraChatWindow(QWidget):
    # Signal to send a message from the chat window to the main logic
    send_message_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        print("AvraChatWindow initializing...")
        self.setWindowTitle("Avra Chat")
        # Use Qt.Dialog flag to make it a separate, top-level window that might block others,
        # or just Qt.Window for a regular non-modal window. Let's stick to Qt.Window
        # with FramelessWindowHint and WindowStaysOnTopHint.
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground) # Enable transparency
        self.setMouseTracking(True) # For dragging this window

        # Set initial size (approx 4x5 aspect, based on 2 ball heights = 300px)
        self.resize(400, 500) # Example size, Width x Height
        self.setMinimumSize(200, 250) # Minimum size for the chat window

        # Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10) # Add some padding
        main_layout.setSpacing(10)

        # Conversation History Display
        self.history_display = QTextEdit()
        self.history_display.setReadOnly(True)
        # Allow vertical expansion but cap at a reasonable height, enable scrolling
        self.history_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.history_display.setMaximumHeight(400) # Max height for history before scrolling
        self.history_display.setStyleSheet(self._get_text_edit_style(is_input=False))
        main_layout.addWidget(self.history_display)

        # User Input Text Edit
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Type your message here...")
        # Allow vertical expansion for typing, but cap it
        self.input_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.input_edit.setMinimumHeight(50) # Minimum input height
        self.input_edit.setMaximumHeight(100) # Cap input height
        self.input_edit.setStyleSheet(self._get_text_edit_style(is_input=True))


        # Install event filter to catch Enter key for sending
        self.input_key_filter = InputKeyFilter()
        self.input_key_filter.enter_pressed.connect(self.send_message)
        self.input_edit.installEventFilter(self.input_key_filter)

        main_layout.addWidget(self.input_edit)

        self.setLayout(main_layout)

        self._drag_position = None # For dragging the chat window

        # Connect to parent's window_moved signal if parent is AvraScreenBot
        if isinstance(self.parent(), AvraScreenBot):
            self.parent().window_moved.connect(self.update_position_from_parent)

        print("AvraChatWindow initialization complete.")

    def _get_text_edit_style(self, is_input: bool):
        """Returns stylesheet for text edits with controlled transparency."""
        bg_alpha = 30 # Alpha for background transparency (adjust as needed, 0-255)
        border_alpha = 60 # Alpha for border transparency for input
        if not is_input:
            border_alpha = 40 # Alpha for border transparency for history

        style = f"""
            QTextEdit {{
                background-color: rgba(255, 255, 255, {bg_alpha});
                border: 1px solid rgba(150, 180, 200, {border_alpha});
                border-radius: 10px;
                padding: 10px; /* Reduced padding slightly for chat */
                color: #000000; /* Black text */
                font-family: "Segoe UI", "Arial", sans-serif;
                font-size: {'12px' if is_input else '14px'};
                outline: none;
            }}
            QTextEdit QScrollBar:vertical {{
                width: 8px; /* Show a slightly wider scrollbar */
            }}
             QTextEdit QScrollBar:horizontal {{
                height: 8px; /* Show a slightly wider scrollbar */
            }}
            QTextEdit QScrollBar::handle:vertical,
            QTextEdit QScrollBar::handle:horizontal {{
                background: rgba(100, 150, 200, 150); /* Semi-transparent blue handle */
                border-radius: 4px; /* Rounded scrollbar handles */
            }}
             QTextEdit QScrollBar::add-line:vertical,
            QTextEdit QScrollBar::sub-line:vertical,
            QTextEdit QScrollBar::add-line:horizontal,
            QTextEdit QScrollBar::sub-line:horizontal {{
                border: none;
                background: none;
            }}
            QTextEdit QScrollBar::add-page:vertical,
            QTextEdit QScrollBar::sub-page:vertical,
             QTextEdit QScrollBar::add-page:horizontal,
            QTextEdit QScrollBar::sub-page:horizontal {{
                background: none;
            }}
             QTextEdit::disabled {{
                background-color: rgba(200, 200, 200, {bg_alpha});
                 color: #666666;
            }}
        """
        if is_input:
             # Specific style for input box placeholder
             style += """
                QTextEdit::placeholder-text {
                    color: rgba(255, 255, 255, 150); /* Lighter and slightly opaque white for placeholder */
                }
             """
        return style


    def send_message(self):
        """Gets text from input, emits signal, and clears input."""
        prompt = self.input_edit.toPlainText().strip()
        if prompt:
            print(f"ChatWindow: Sending message: '{prompt[:50]}...'")
            self.send_message_requested.emit(prompt) # Emit signal
            self.input_edit.clear() # Clear input box
            # Reset input height after clearing text
            self.input_edit.setMinimumHeight(50)
            self.input_edit.setMaximumHeight(100) # Re-apply max height in case it was altered
            self.input_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) # Fixed height for input after sending
            self.input_edit.verticalScrollBar().setValue(0) # Scroll input back to top


    def append_message(self, role: str, text: str, is_user: bool = False):
        """Appends a message (user or Avra) to the history display."""
        # Format message with bold role and regular text
        role_color = "#000000" # Black for Avra
        if is_user:
            role_color = "#222222" # Slightly grey for User

        formatted_text = f"<b style='color:{role_color};'>{role.capitalize()}:</b> {text.strip()}<br><br>"
        self.history_display.insertHtml(formatted_text) # Use insertHtml for formatting

        # Scroll to the bottom smoothly (optional animation)
        # self._animate_scroll_to_bottom()
        self.history_display.verticalScrollBar().setValue(self.history_display.verticalScrollBar().maximum())

        print(f"ChatWindow: Appended message from {role}.")

    def display_thinking_state(self):
           """Appends a thinking indicator to the history."""
           # Use a unique tag or identifier to easily find and replace this later
           thinking_html = "<span id='thinking_indicator'><b>Avra:</b> <i>Thinking...</i></span>" # Removed trailing line breaks
           self.history_display.insertHtml(thinking_html)
           self.history_display.verticalScrollBar().setValue(self.history_display.verticalScrollBar().maximum())


    def display_streaming_chunk(self, chunk: str):
        """Appends a streaming chunk to the last message (Avra's thinking message)."""
        # Find the thinking indicator
        history_html = self.history_display.toHtml()
        # Using a more robust search for the span tag and its content
        thinking_indicator_tag = "<span id=\"thinking_indicator\">"
        thinking_indicator_start = history_html.rfind(thinking_indicator_tag) # Use rfind to get the last occurrence

        if thinking_indicator_start != -1:
            # Move cursor to the end of the thinking indicator span
            cursor = self.history_display.textCursor()
            # Find the position right after the closing span tag of the thinking indicator
            pos_after_tag = history_html.find("</span>", thinking_indicator_start) + len("</span>")
            if pos_after_tag != -1:
                cursor.setPosition(pos_after_tag)
                self.history_display.setTextCursor(cursor)

                # Now the cursor is positioned after the span tag. Insert the chunk.
                self.history_display.insertPlainText(chunk)
            else:
                 # Fallback if closing span tag is not found
                 cursor.movePosition(cursor.End)
                 self.history_display.setTextCursor(cursor)
                 self.history_display.insertPlainText(chunk)

        else:
             # If thinking indicator wasn't found (e.g., first chunk after an error message)
             # Just append the chunk directly.
             cursor = self.history_display.textCursor()
             cursor.movePosition(cursor.End)
             self.history_display.setTextCursor(cursor)
             self.history_display.insertPlainText(chunk)


        # Scroll to the bottom after inserting chunk
        self.history_display.verticalScrollBar().setValue(self.history_display.verticalScrollBar().maximum())


    def end_streaming_display(self):
        """Called when streaming finishes to finalize the last message."""
        # Check if the last few characters are part of the "Thinking..." state and remove it
        history_html = self.history_display.toHtml()
        thinking_indicator_tag = "<span id=\"thinking_indicator\">"
        thinking_indicator_start = history_html.rfind(thinking_indicator_tag)

        if thinking_indicator_start != -1:
             # Find the closing span tag for the thinking indicator
             thinking_indicator_end = history_html.find("</span>", thinking_indicator_start)
             if thinking_indicator_end != -1:
                 # Select the entire span tag including its content
                 cursor = self.history_display.textCursor()
                 cursor.setPosition(thinking_indicator_start, cursor.MoveAnchor)
                 cursor.setPosition(thinking_indicator_end + len("</span>"), cursor.KeepAnchor)
                 cursor.removeSelectedText() # Remove the "Thinking..." part

        # Ensure the last message ends with line breaks if needed
        history_plain_text = self.history_display.toPlainText()
        if not history_plain_text.strip().endswith("\n"): # Check for at least one newline after stripping whitespace
              self.history_display.append("") # Add a new paragraph (<p>), which often renders as two line breaks

        self.history_display.verticalScrollBar().setValue(self.history_display.verticalScrollBar().maximum())
        print("ChatWindow: Streaming display finalized.")


    # Methods for dragging the chat window
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for window dragging."""
        if event.button() == Qt.LeftButton:
            # Only start drag if clicking on the frame, not the text edits
            # Check if the click is within the window area but not on a child widget that consumes events
            child = self.childAt(event.pos())
            # Allow dragging from the transparent background/border area
            if child is None or (isinstance(child, (QTextEdit,)) and not child.geometry().contains(event.pos())):
                 self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
                 event.accept()
            else:
                 super().mousePressEvent(event) # Let other child widgets handle their events

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for window dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_position is not None:
            self.move(event.globalPos() - self._drag_position)
            event.accept()
        else:
             super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release for window dragging."""
        if event.button() == Qt.LeftButton and self._drag_position is not None:
            self._drag_position = None
            event.accept()
        else:
             super().mouseReleaseEvent(event)

    def update_position_from_parent(self, parent_global_pos: QPoint):
        """
        Slot to update the chat window's position based on the parent's new global position.
        Requires the parent (AvraScreenBot) to store the relative offset.
        """
        if isinstance(self.parent(), AvraScreenBot) and self.parent()._chat_window_relative_offset is not None:
            new_chat_global_pos = parent_global_pos + self.parent()._chat_window_relative_offset
            self.move(new_chat_global_pos)
            # print(f"ChatWindow: Moved to {new_chat_global_pos.x()}, {new_chat_global_pos.y()} based on parent move.") # Optional debug


    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create and show a context menu on right-click."""
        context_menu = QMenu(self)
        close_action = QAction("Close Chat", self)
        close_action.triggered.connect(self.hide) # Hide the chat window
        context_menu.addAction(close_action)
        context_menu.exec_(QCursor.pos())


# --- Main Application Window (Borderless, Transparent, Draggable) ---

class AvraScreenBot(QWidget):
    window_moved = pyqtSignal(QPoint) # Signal to indicate main window has moved

    def __init__(self):
        super().__init__()
        print("AvraScreenBot initializing...")
        self.setWindowTitle("Avra Screen Bot")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)

        # Main layout only needs to hold the circle
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0) # No spacing needed in the main window now

        # Gradient Circle Widget instance
        self.circle_widget = GradientCircleWidget()
        # Connect to the slot that shows the chat window
        self.circle_widget.circle_double_clicked.connect(self.toggle_chat_window)
        main_layout.addWidget(self.circle_widget, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

        self._drag_position = None # To store the position when dragging starts
        self._chat_window_relative_offset = QPoint() # To store offset for chat window movement

        # --- Chat Window and Gemini Session ---
        # Create the chat window, parented to main window for signal access
        self.chat_window = AvraChatWindow(self)
        # Connect chat window signal to main bot processing slot
        self.chat_window.send_message_requested.connect(self.process_user_prompt)
        # Connect the chat thread's finished signal to a cleanup slot in the main window
        # (This was missing, adding it now for proper thread management)
        # Need to connect this after the thread is created, so it's done in process_user_prompt
        # self.chat_window.gemini_thread.finished.connect(self.on_gemini_chat_thread_finished) # Cannot connect here

        # Initialize Gemini Client and Chat Session
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                 raise ValueError("GEMINI_API_KEY environment variable not set.")
            self.gemini_client = genai.Client(api_key=api_key)
            self.chat_session = None # Will be created on first message interaction
            print("AvraScreenBot: Gemini Client initialized.")
        except Exception as e:
            print(f"AvraScreenBot: Failed to initialize Gemini Client: {e}")
            # Handle the error - perhaps disable chat functionality or show a warning
            self.gemini_client = None
            self.chat_session = None
            # Display error in the chat window if it's somehow already visible,
            # or when it's next opened.
            self.chat_window.append_message("Avra", f"Initialization Error: Could not connect to AI service. {e}")


        self.gemini_thread = None # Initialize thread attribute

        # --- Setup Keyboard Shortcut ---
        # Connect Ctrl+Space to toggling the chat window
        self.shortcut_toggle_chat = QShortcut(QKeySequence("Ctrl+Space"), self)
        self.shortcut_toggle_chat.activated.connect(self.toggle_chat_window)
        print("Keyboard shortcut 'Ctrl+Space' configured to toggle chat window.")
        # -------------------------------

        print("AvraScreenBot initialization complete.")

    def toggle_chat_window(self):
        """Shows or hides the chat window."""
        if self.chat_window.isVisible():
            print("AvraScreenBot: Hiding chat window.")
            self.chat_window.hide()
        else:
            print("AvraScreenBot: Showing chat window.")
            # Position relative to the circle (Approximation of "growing from")
            # Get the center of the circle in global screen coordinates
            circle_center_global = self.circle_widget.mapToGlobal(self.circle_widget.rect().center())
            chat_window_rect = self.chat_window.rect()

            # Calculate the top-left position for the chat window
            # So its top-center is aligned with the circle's bottom-center, plus a small offset
            circle_bottom_global_y = self.circle_widget.mapToGlobal(self.circle_widget.rect().bottomLeft()).y()
            circle_center_global_x = circle_center_global.x()

            # Position the top-left of the chat window
            chat_pos_x = circle_center_global_x - (chat_window_rect.width() // 2)
            chat_pos_y = circle_bottom_global_y + 10 # Add 10 pixels spacing below the circle

            # Move the chat window to the calculated global position
            self.chat_window.move(chat_pos_x, chat_pos_y)

            # Store the relative offset *after* moving the chat window for the first time
            # This offset is between the main window's top-left and the chat window's top-left
            self._chat_window_relative_offset = self.chat_window.pos() - self.pos()

            self.chat_window.show()
            self.chat_window.input_edit.setFocus() # Give focus to the chat input


    # Mouse event handling for the main window (dragging the circle)
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for window dragging."""
        if event.button() == Qt.LeftButton:
            # Only start drag if clicking on the circle widget or the transparent background
            child = self.childAt(event.pos())
            if child is None or isinstance(child, GradientCircleWidget):
                self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()
            else:
                super().mousePressEvent(event) # Let other child widgets handle their events


    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for window dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_position is not None:
            self.move(event.globalPos() - self._drag_position)
            # Emit the new global position of the main window
            self.window_moved.emit(self.pos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release for window dragging."""
        if event.button() == Qt.LeftButton and self._drag_position is not None:
            self._drag_position = None
            event.accept()
        else:
             super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create and show a context menu on right-click."""
        context_menu = QMenu(self)
        # Add action to toggle chat window
        toggle_chat_action = QAction("Toggle Chat Window (Ctrl+Space)", self)
        toggle_chat_action.triggered.connect(self.toggle_chat_window)
        context_menu.addAction(toggle_chat_action)

        close_action = QAction("Close Avra", self)
        close_action.triggered.connect(self.close) # Connect to the window's close method
        context_menu.addAction(close_action)

        context_menu.exec_(QCursor.pos())


    # --- Prompt Processing for Chat ---
    def process_user_prompt(self, prompt: str):
        """Handles the user's prompt using the chat session."""
        print(f"AvraScreenBot: Processing chat prompt: '{prompt[:50]}...'")

        if self.gemini_thread is not None and self.gemini_thread.isRunning():
             print("AvraScreenBot: Gemini thread already running, waiting...")
             self.chat_window.append_message("Avra", "I'm still processing the previous request, please wait.")
             return # Don't send a new message if one is in progress

        if self.gemini_client is None:
             print("AvraScreenBot: Gemini Client not initialized due to previous error.")
             self.chat_window.append_message("Avra", "Error: Avra is unable to connect to the AI service.")
             return

        # If chat session doesn't exist, create it
        if self.chat_session is None:
             try:
                 self.chat_session = self.gemini_client.chats.create(model="gemini-2.5-flash-preview-04-17")
                 print("AvraScreenBot: New chat session created.")
             except Exception as e:
                  print(f"AvraScreenBot: Failed to create chat session: {e}")
                  self.chat_window.append_message("Avra", f"Error: Could not start chat session: {e}")
                  return

        # Display user message in chat history
        self.chat_window.append_message("You", prompt, is_user=True) # Mark as user message
        self.chat_window.display_thinking_state() # Show thinking indicator

        # Start the Gemini API call in a new thread
        self.gemini_thread = GeminiChatThread(self.chat_session, prompt)
        # Connect signals from the thread to update the chat window
        self.gemini_thread.response_chunk_signal.connect(self.chat_window.display_streaming_chunk)
        # Connect the finished signal to both finalizing display and cleanup slot
        self.gemini_thread.response_finished_signal.connect(self.chat_window.end_streaming_display)
        self.gemini_thread.response_finished_signal.connect(self.on_gemini_chat_thread_finished)
        self.gemini_thread.error_signal.connect(self.on_gemini_chat_error)

        self.gemini_thread.start() # Start the thread
        print("AvraScreenBot: GeminiChatThread started.")

    def on_gemini_chat_thread_finished(self):
        """Slot called when the Gemini chat thread finishes streaming."""
        print("AvraScreenBot: on_gemini_chat_thread_finished received signal.")
        # The chat window's end_streaming_display slot is connected, it handles finalization.
        # Clean up the thread reference
        self.gemini_thread = None
        print("AvraScreenBot: Gemini chat thread reference cleared.")
        # Restore focus to the input box after response is finalized
        # Check if chat window is still visible before setting focus
        if self.chat_window.isVisible():
             self.chat_window.input_edit.setFocus()


    def on_gemini_chat_error(self, error_message: str):
           """Slot to handle errors from the Gemini chat thread."""
           print(f"AvraScreenBot: on_gemini_chat_error received signal: {error_message}")
           # Append the error message to the chat history
           self.chat_window.append_message("Avra", f"Error: {error_message}")
           # Clean up the thread reference
           self.gemini_thread = None
           print("AvraScreenBot: Gemini chat thread reference cleared after error.")
           # Restore focus to the input box after an error
           # Check if chat window is still visible before setting focus
           if self.chat_window.isVisible():
                self.chat_window.input_edit.setFocus()


# --- Main Application Entry Point ---

if __name__ == "__main__":
    # Check for API key before starting the application
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set the GEMINI_API_KEY and run the script again.")
        sys.exit(1) # Exit with an error code

    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Set a modern style for the application

    # Create and show the main window (just the circle initially)
    avra_screen_bot = AvraScreenBot()
    avra_screen_bot.show()

    # Start the application event loop
    sys.exit(app.exec_())