import sqlite3

conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# All students
cursor.execute("SELECT * FROM students")
students = cursor.fetchall()
print("Students:", students)

# All attendance
cursor.execute("SELECT * FROM attendance") 
attendance = cursor.fetchall()
print("Attendance:", attendance)

conn.close()