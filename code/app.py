import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

from detector import detect_and_decode_all_codes


class CodeScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Barcode + QR Scanner")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f4f4f4")

        self.image_path = None
        self.input_image_np = None
        self.result_image_np = None

        title = tk.Label(
            root,
            text="Unified Barcode + QR Scanner",
            font=("Arial", 20, "bold"),
            bg="#f4f4f4"
        )
        title.pack(pady=10)

        button_frame = tk.Frame(root, bg="#f4f4f4")
        button_frame.pack(pady=10)

        self.upload_btn = tk.Button(
            button_frame,
            text="Upload Image",
            font=("Arial", 12),
            width=15,
            command=self.upload_image
        )
        self.upload_btn.grid(row=0, column=0, padx=10)

        self.scan_btn = tk.Button(
            button_frame,
            text="Scan",
            font=("Arial", 12, "bold"),
            width=15,
            bg="#4CAF50",
            fg="white",
            command=self.scan_image
        )
        self.scan_btn.grid(row=0, column=1, padx=10)

        self.images_frame = tk.Frame(root, bg="#f4f4f4")
        self.images_frame.pack(pady=10)

        self.original_label_title = tk.Label(
            self.images_frame,
            text="Original Image",
            font=("Arial", 13, "bold"),
            bg="#f4f4f4"
        )
        self.original_label_title.grid(row=0, column=0, padx=20, pady=5)

        self.result_label_title = tk.Label(
            self.images_frame,
            text="Detected / Decoded Result",
            font=("Arial", 13, "bold"),
            bg="#f4f4f4"
        )
        self.result_label_title.grid(row=0, column=1, padx=20, pady=5)

        self.original_image_label = tk.Label(
            self.images_frame,
            bg="white",
            width=450,
            height=300,
            relief="solid",
            bd=1
        )
        self.original_image_label.grid(row=1, column=0, padx=20, pady=10)

        self.result_image_label = tk.Label(
            self.images_frame,
            bg="white",
            width=450,
            height=300,
            relief="solid",
            bd=1
        )
        self.result_image_label.grid(row=1, column=1, padx=20, pady=10)

        self.results_text = tk.Text(
            root,
            height=14,
            width=130,
            font=("Consolas", 11)
        )
        self.results_text.pack(padx=20, pady=15)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.image_path = file_path

        pil_img = Image.open(file_path).convert("RGB")
        self.input_image_np = np.array(pil_img)

        self.display_image(pil_img, self.original_image_label)

        self.result_image_label.config(image="")
        self.result_image_label.image = None

        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, f"Image loaded:\n{file_path}\n")

    def scan_image(self):
        if self.input_image_np is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        try:
            final_img, decoded_barcodes, decoded_qrs, debug = detect_and_decode_all_codes(self.input_image_np)
            self.result_image_np = final_img

            result_pil = Image.fromarray(final_img.astype(np.uint8))
            self.display_image(result_pil, self.result_image_label)

            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, "=== BARCODE RESULTS ===\n")

            if len(decoded_barcodes) == 0:
                self.results_text.insert(tk.END, "No barcode detected.\n")
            else:
                for i, item in enumerate(decoded_barcodes, 1):
                    self.results_text.insert(tk.END, f"\nBarcode {i}\n")
                    self.results_text.insert(tk.END, f"Box     : {item['box']}\n")
                    self.results_text.insert(tk.END, f"Decoded : {item['decoded']}\n")
                    self.results_text.insert(tk.END, f"Text    : {item['text']}\n")
                    self.results_text.insert(tk.END, f"Type    : {item['type']}\n")

            self.results_text.insert(tk.END, "\n=== QR RESULTS ===\n")

            if len(decoded_qrs) == 0:
                self.results_text.insert(tk.END, "No QR code detected.\n")
            else:
                for i, item in enumerate(decoded_qrs, 1):
                    self.results_text.insert(tk.END, f"\nQR {i}\n")
                    self.results_text.insert(tk.END, f"Box     : {item['box']}\n")
                    self.results_text.insert(tk.END, f"Decoded : {item['decoded']}\n")
                    self.results_text.insert(tk.END, f"Text    : {item['text']}\n")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")

    def display_image(self, pil_img, label):
        preview = pil_img.copy()
        preview.thumbnail((450, 300))

        tk_img = ImageTk.PhotoImage(preview)
        label.config(image=tk_img)
        label.image = tk_img


if __name__ == "__main__":
    root = tk.Tk()
    app = CodeScannerApp(root)
    root.mainloop()
