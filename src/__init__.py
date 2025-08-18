"""
Student Management System Package

This package provides a complete student management system with support for
multiple databases and basic CRUD operations on student records.

Modules
-------
school
    Contains the core Student and StudentManagers classes
main
    Provides the command-line interface for the application

Classes
-------
Student
    Represents a student with name and surname
StudentManagers
    Manages multiple student databases

Examples
--------
>>> from src.school import Student, StudentManagers
>>> sm = StudentManagers()
>>> sm.create_db("my_class")
>>> sm.set_current_db("my_class")
>>> student = Student("Alice", "Cooper")
>>> sm.add_student(student)
"""