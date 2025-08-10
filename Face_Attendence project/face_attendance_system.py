import cv2
import os
import numpy as np
import sqlite3
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading

class FaceAttendanceSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Detection Attendance System")
        self.root.geometry("800x600")
        
        # Initialize OpenCV components
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Create directories
        self.create_directories()
        
        # Initialize database
        self.init_database()
        
        # Create GUI
        self.create_gui()
        
        # Load trained model if exists
        self.load_model()
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs('dataset', exist_ok=True)
        os.makedirs('trainer', exist_ok=True)
        
    def init_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect('attendance.db')
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                date TEXT,
                time TEXT,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        ''')
        
        self.conn.commit()
        
    def create_gui(self):
        """Create the main GUI"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Registration Tab
        reg_frame = ttk.Frame(notebook)
        notebook.add(reg_frame, text="Register Face")
        self.create_registration_tab(reg_frame)
        
        # Attendance Tab
        att_frame = ttk.Frame(notebook)
        notebook.add(att_frame, text="Take Attendance")
        self.create_attendance_tab(att_frame)
        
        # View Records Tab
        view_frame = ttk.Frame(notebook)
        notebook.add(view_frame, text="View Records")
        self.create_view_tab(view_frame)
        
    def create_registration_tab(self, parent):
        """Create registration interface"""
        # Input fields
        ttk.Label(parent, text="Student ID:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.id_entry = ttk.Entry(parent, width=20)
        self.id_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(parent, text="Name:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.name_entry = ttk.Entry(parent, width=20)
        self.name_entry.grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Label(parent, text="Email:").grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.email_entry = ttk.Entry(parent, width=20)
        self.email_entry.grid(row=2, column=1, padx=10, pady=10)
        
        # Buttons
        ttk.Button(parent, text="Capture Faces", command=self.capture_faces).grid(row=3, column=0, padx=10, pady=20)
        ttk.Button(parent, text="Train Model", command=self.train_model).grid(row=3, column=1, padx=10, pady=20)
        
        # Status label
        self.reg_status = ttk.Label(parent, text="Ready to register faces")
        self.reg_status.grid(row=4, column=0, columnspan=2, pady=10)
        
    def create_attendance_tab(self, parent):
        """Create attendance interface"""
        # Video frame
        self.video_frame = ttk.Frame(parent)
        self.video_frame.pack(pady=20)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(pady=20)
        
        self.start_btn = ttk.Button(button_frame, text="Start Attendance", command=self.start_attendance)
        self.start_btn.pack(side='left', padx=10)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Attendance", command=self.stop_attendance)
        self.stop_btn.pack(side='left', padx=10)
        
        # Status
        self.att_status = ttk.Label(parent, text="Click 'Start Attendance' to begin")
        self.att_status.pack(pady=10)
        
        # Attendance list
        self.att_tree = ttk.Treeview(parent, columns=('Name', 'Time'), show='headings', height=8)
        self.att_tree.heading('Name', text='Name')
        self.att_tree.heading('Time', text='Time')
        self.att_tree.pack(pady=20, fill='both', expand=True)
        
    def create_view_tab(self, parent):
        """Create view records interface"""
        # Date selection
        date_frame = ttk.Frame(parent)
        date_frame.pack(pady=10)
        
        ttk.Label(date_frame, text="Select Date:").pack(side='left', padx=5)
        self.date_entry = ttk.Entry(date_frame, width=15)
        self.date_entry.pack(side='left', padx=5)
        self.date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        ttk.Button(date_frame, text="View Records", command=self.view_records).pack(side='left', padx=10)
        ttk.Button(date_frame, text="Export CSV", command=self.export_csv).pack(side='left', padx=5)
        
        # Records tree
        self.records_tree = ttk.Treeview(parent, columns=('ID', 'Name', 'Date', 'Time'), show='headings', height=15)
        self.records_tree.heading('ID', text='Student ID')
        self.records_tree.heading('Name', text='Name')
        self.records_tree.heading('Date', text='Date')
        self.records_tree.heading('Time', text='Time')
        self.records_tree.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=self.records_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.records_tree.configure(yscrollcommand=scrollbar.set)
        
    def capture_faces(self):
        """Capture face samples for training"""
        try:
            student_id = int(self.id_entry.get())
            name = self.name_entry.get()
            email = self.email_entry.get()
            
            if not name:
                messagebox.showerror("Error", "Please enter student name")
                return
                
            # Save student info to database
            self.cursor.execute("INSERT OR REPLACE INTO students (id, name, email) VALUES (?, ?, ?)",
                              (student_id, name, email))
            self.conn.commit()
            
            # Start face capture
            self.reg_status.config(text="Starting camera...")
            self.root.update()
            
            cap = cv2.VideoCapture(0)
            count = 0
            
            self.reg_status.config(text="Capturing faces... Look at camera")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    count += 1
                    
                    # Save face sample
                    cv2.imwrite(f"dataset/User.{student_id}.{count}.jpg", gray[y:y+h, x:x+w])
                    
                cv2.putText(frame, f"Samples: {count}/50", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Capture', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            
            self.reg_status.config(text=f"Captured {count} samples for {name}")
            messagebox.showinfo("Success", f"Face samples captured for {name}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid student ID")
        except Exception as e:
            messagebox.showerror("Error", f"Error capturing faces: {str(e)}")
            
    def train_model(self):
        """Train the face recognition model"""
        try:
            self.reg_status.config(text="Training model...")
            self.root.update()
            
            path = 'dataset'
            faces = []
            ids = []
            
            for filename in os.listdir(path):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(path, filename)
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img, 'uint8')
                    
                    # Extract ID from filename
                    student_id = int(filename.split('.')[1])
                    
                    faces.append(img_array)
                    ids.append(student_id)
                    
            if faces:
                self.recognizer.train(faces, np.array(ids))
                self.recognizer.write('trainer/trainer.yml')
                
                self.reg_status.config(text="Model trained successfully!")
                messagebox.showinfo("Success", "Face recognition model trained successfully!")
            else:
                messagebox.showwarning("Warning", "No face samples found. Please capture faces first.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
            
    def load_model(self):
        """Load trained model if exists"""
        try:
            if os.path.exists('trainer/trainer.yml'):
                self.recognizer.read('trainer/trainer.yml')
        except:
            pass
            
    def start_attendance(self):
        """Start attendance taking"""
        try:
            self.attendance_running = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            
            # Clear previous attendance for today
            today = datetime.now().strftime("%Y-%m-%d")
            for item in self.att_tree.get_children():
                self.att_tree.delete(item)
                
            self.att_status.config(text="Attendance system running...")
            
            # Start video capture in separate thread
            self.video_thread = threading.Thread(target=self.video_capture_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error starting attendance: {str(e)}")
            
    def stop_attendance(self):
        """Stop attendance taking"""
        self.attendance_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.att_status.config(text="Attendance stopped")
        
    def video_capture_loop(self):
        """Video capture loop for attendance"""
        # Create separate database connection for this thread
        thread_conn = sqlite3.connect('attendance.db')
        thread_cursor = thread_conn.cursor()
        
        cap = cv2.VideoCapture(0)
        marked_today = set()
        
        while self.attendance_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Recognize face
                roi_gray = gray[y:y+h, x:x+w]
                id_, confidence = self.recognizer.predict(roi_gray)
                
                if confidence < 70:  # Confidence threshold
                    # Get student name using thread connection
                    thread_cursor.execute("SELECT name FROM students WHERE id = ?", (id_,))
                    result = thread_cursor.fetchone()
                    
                    if result and id_ not in marked_today:
                        name = result[0]
                        now = datetime.now()
                        date = now.strftime("%Y-%m-%d")
                        time = now.strftime("%H:%M:%S")
                        
                        # Mark attendance using thread connection
                        thread_cursor.execute("INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
                                            (id_, date, time))
                        thread_conn.commit()
                        
                        # Add to GUI using thread-safe method
                        self.root.after(0, lambda n=name, t=time: self.att_tree.insert('', 'end', values=(n, t)))
                        marked_today.add(id_)
                        
                        cv2.putText(frame, f"{name} - Marked", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    elif result:
                        name = result[0]
                        cv2.putText(frame, f"{name} - Already marked", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            # Convert frame for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((400, 300))
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update GUI using thread-safe method
            self.root.after(0, lambda img=frame_tk: self.update_video_label(img))
            
        cap.release()
        thread_conn.close()
        
    def view_records(self):
        """View attendance records for selected date"""
        try:
            date = self.date_entry.get()
            
            # Clear previous records
            for item in self.records_tree.get_children():
                self.records_tree.delete(item)
                
            # Fetch records
            self.cursor.execute("""
                SELECT s.id, s.name, a.date, a.time 
                FROM attendance a 
                JOIN students s ON a.student_id = s.id 
                WHERE a.date = ? 
                ORDER BY a.time
            """, (date,))
            
            records = self.cursor.fetchall()
            
            for record in records:
                self.records_tree.insert('', 'end', values=record)
                
            if not records:
                messagebox.showinfo("Info", f"No attendance records found for {date}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error viewing records: {str(e)}")
            
    def export_csv(self):
        """Export attendance records to CSV"""
        try:
            import csv
            
            date = self.date_entry.get()
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialname=f"attendance_{date}.csv"
            )
            
            if filename:
                self.cursor.execute("""
                    SELECT s.id, s.name, a.date, a.time 
                    FROM attendance a 
                    JOIN students s ON a.student_id = s.id 
                    WHERE a.date = ? 
                    ORDER BY a.time
                """, (date,))
                
                records = self.cursor.fetchall()
                
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Student ID', 'Name', 'Date', 'Time'])
                    writer.writerows(records)
                    
                messagebox.showinfo("Success", f"Records exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting CSV: {str(e)}")
            
    def update_video_label(self, frame_tk):
        """Thread-safe method to update video label"""
        self.video_label.configure(image=frame_tk)
        self.video_label.image = frame_tk
        
    def run(self):
        """Start the application"""
        self.root.mainloop()
        self.conn.close()

if __name__ == "__main__":
    # Create and run the application
    app = FaceAttendanceSystem()
    app.run()