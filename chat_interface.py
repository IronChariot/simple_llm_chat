import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import json
from ollama import chat_completion, get_available_models
import threading

class ChatInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ollama Chat Interface")
        self.root.geometry("600x800")

        self.conversation = []
        self.setup_ui()

    def setup_ui(self):
        # Model selection frame
        model_frame = ttk.Frame(self.root)
        model_frame.pack(pady=5, fill=tk.X)

        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=(10, 5))
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, width=20)
        self.model_dropdown.pack(side=tk.LEFT, padx=5)
        self.update_model_list()

        # System Prompt area
        system_frame = ttk.Frame(self.root)
        system_frame.pack(pady=5, fill=tk.X)

        ttk.Label(system_frame, text="System Prompt:").pack(side=tk.LEFT, padx=(10, 5))
        self.system_prompt = scrolledtext.ScrolledText(system_frame, wrap=tk.WORD, height=3)
        self.system_prompt.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Temperature input
        ttk.Label(model_frame, text="Temp:").pack(side=tk.LEFT, padx=(10, 5))
        self.temperature_var = tk.StringVar(value="0.0")
        self.temperature_entry = ttk.Entry(model_frame, textvariable=self.temperature_var, width=5)
        self.temperature_entry.pack(side=tk.LEFT, padx=5)

        # Max Tokens input
        ttk.Label(model_frame, text="Max Tokens:").pack(side=tk.LEFT, padx=(10, 5))
        self.max_tokens_var = tk.StringVar(value="4000")
        self.max_tokens_entry = ttk.Entry(model_frame, textvariable=self.max_tokens_var, width=7)
        self.max_tokens_entry.pack(side=tk.LEFT, padx=5)

        # Clear/Save/Load conversation
        frame = ttk.Frame(self.root)
        frame.pack(pady=5)
        ttk.Button(frame, text="Clear", command=self.clear_conversation).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Save", command=self.save_conversation).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Load", command=self.load_conversation).pack(side=tk.LEFT)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=37)
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # Configure text tags for coloring
        self.chat_display.tag_configure("user", foreground="dark red")
        self.chat_display.tag_configure("ai", foreground="dark green")

        # User input (ScrolledText widget)
        self.user_input = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=3, width=50)
        self.user_input.pack(padx=10, pady=5, fill=tk.X, expand=True)

        # Modify the Send button to use the new send_message_async method
        ttk.Button(self.root, text="Send", command=self.send_message_async).pack(pady=5)

        # Bind the Return key to send_message_async, but allow Shift+Return for new lines
        self.user_input.bind("<Return>", self.handle_return)

    def handle_return(self, event):
        if not event.state & 0x1:  # Check if Shift is not pressed
            return self.send_message_async()
        return None  # Allow default behavior for Shift+Return

    def send_message_async(self, event=None):
        user_message = self.user_input.get("1.0", tk.END).strip()
        if user_message:
            self.add_message("user", user_message)
            self.user_input.delete("1.0", tk.END)
            self.user_input.config(state=tk.DISABLED)  # Disable input field

            # Start a new thread for AI response
            threading.Thread(target=self.get_ai_response, daemon=True).start()

        return "break"  # Prevents the default behavior of Return key

    def get_ai_response(self):
        model = self.model_var.get()
        messages = self.prepare_conversation_history()
        
        # Get temperature and max tokens values
        try:
            temperature = float(self.temperature_var.get())
            temperature = max(0.0, min(1.0, temperature))  # Clamp between 0.0 and 1.0
        except ValueError:
            temperature = 0.0

        try:
            max_tokens = int(self.max_tokens_var.get())
            max_tokens = max(1, max_tokens)  # Ensure it's at least 1
        except ValueError:
            max_tokens = 4000

        # Start a new thread for AI response
        threading.Thread(target=self.stream_ai_response, args=(model, messages, temperature, max_tokens), daemon=True).start()

    def stream_ai_response(self, model, messages, temperature, max_tokens):
        full_response = ""
        is_first_token = True
        for chunk in chat_completion(model, messages=messages, temperature=temperature, max_tokens=max_tokens):
            if 'message' in chunk:
                content = chunk['message'].get('content', '')
                full_response += content
                # Use after method to update UI from the main thread
                self.root.after(0, self.update_chat_with_stream, content, is_first_token)
                is_first_token = False
            if chunk.get('done', False):
                break
        
        # Add the full response to the conversation history
        self.conversation.append({"role": "assistant", "content": full_response})
        
        # Add newlines after the last token
        self.root.after(0, self.finalize_ai_response)

    def update_chat_with_stream(self, content, is_first_token):
        self.chat_display.config(state=tk.NORMAL)
        if is_first_token:
            self.chat_display.insert(tk.END, "AI:\n", "ai")
        self.chat_display.insert(tk.END, content, "ai")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def finalize_ai_response(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n\n", "ai")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.enable_input()

    def enable_input(self):
        self.user_input.config(state=tk.NORMAL)
        self.user_input.focus_set()

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation.append(message)
        self.chat_display.config(state=tk.NORMAL)
        tag = "user" if role == "user" else "ai"
        sender = "You" if role == "user" else "AI"
        self.chat_display.insert(tk.END, f"{sender}:\n", tag)
        self.chat_display.insert(tk.END, f"{content}\n\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def update_model_list(self):
        models = get_available_models()
        self.model_dropdown['values'] = models
        if models and not self.model_var.get():
            self.model_var.set(models[0])

    def prepare_conversation_history(self):
        system_message = self.system_prompt.get("1.0", tk.END).strip()
        if system_message:
            return [{"role": "system", "content": system_message}] + self.conversation
        return self.conversation

    def save_conversation(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        if file_path:
            system_message = self.system_prompt.get("1.0", tk.END).strip()
            data_to_save = {
                "system_prompt": system_message,
                "conversation": self.conversation
            }
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f)

    def load_conversation(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.conversation = data.get("conversation", [])
            system_prompt = data.get("system_prompt", "")
            
            self.system_prompt.delete(1.0, tk.END)
            self.system_prompt.insert(tk.END, system_prompt)
            
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            for msg in self.conversation:
                tag = "user" if msg['role'] == "user" else "ai"
                sender = "You" if msg['role'] == "user" else "AI"
                self.chat_display.insert(tk.END, f"{sender}:\n", tag)
                self.chat_display.insert(tk.END, f"{msg['content']}\n\n", tag)
            self.chat_display.config(state=tk.DISABLED)

    def clear_conversation(self):
        self.conversation = []
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.system_prompt.delete(1.0, tk.END)

    def run(self):
        self.root.mainloop()