"""
Student Management System

This module provides classes for managing student data across multiple databases.
It includes functionality for creating students, managing multiple databases,
and performing operations like adding students and displaying them in sorted order.

Classes
-------
Student
    Represents a student with name and surname, with comparison operations
StudentManagers
    Manages multiple student databases with CRUD operations

Examples
--------
>>> from school import Student, StudentManagers
>>> sm = StudentManagers()
>>> sm.create_db("math_class")
>>> sm.set_current_db("math_class")
>>> student = Student("Alice", "Johnson")
>>> sm.add_student(student)
>>> sm.show_students()
Name: Alice, Surname: Johnson
"""


class Student:
    """
    A class representing a student with name and surname.
    
    The Student class provides basic functionality for managing student data,
    including comparison operations for sorting and equality checking.
    
    Parameters
    ----------
    name : str
        The first name of the student
    surname : str
        The last name of the student
        
    Attributes
    ----------
    name__ : str
        The first name of the student (private attribute)
    surname__ : str
        The last name of the student (private attribute)
        
    Examples
    --------
    >>> student = Student("John", "Doe")
    >>> print(student)
    Name: John, Surname: Doe
    """
    
    def __init__(self, name: str, surname: str):
        """
        Initialize a new Student instance.
        
        Parameters
        ----------
        name : str
            The first name of the student
        surname : str
            The last name of the student
        """
        self.name__ = name
        self.surname__ = surname

    def __eq__(self, other) -> bool:
        """
        Check if two Student instances are equal.
        
        Two students are considered equal if they have the same name and surname.
        
        Parameters
        ----------
        other : object
            The other object to compare with this Student instance
            
        Returns
        -------
        bool
            True if both students have the same name and surname, False otherwise
            
        Examples
        --------
        >>> student1 = Student("John", "Doe")
        >>> student2 = Student("John", "Doe")
        >>> student1 == student2
        True
        """
        if not isinstance(other, Student):
            return False
        return self.name__ == other.name__ and self.surname__ == other.surname__

    def __lt__(self, other) -> bool:
        """
        Check if this student is less than another student for sorting purposes.
        
        Students are compared first by surname, then by name (lexicographically).
        
        Parameters
        ----------
        other : Student
            The other Student instance to compare with
            
        Returns
        -------
        bool
            True if this student should be sorted before the other student, False otherwise
        NotImplemented
            If the other object is not a Student instance
            
        Examples
        --------
        >>> student1 = Student("Alice", "Brown")
        >>> student2 = Student("Bob", "Anderson")
        >>> student1 < student2
        False
        """
        if not isinstance(other, Student):
            return NotImplemented
        return (self.surname__, self.name__) < (other.surname__, other.name__)

    def __repr__(self):
        """
        Return a string representation of the Student instance.
        
        Returns
        -------
        str
            A formatted string containing the student's name and surname
            
        Examples
        --------
        >>> student = Student("John", "Doe")
        >>> repr(student)
        'Name: John, Surname: Doe'
        """
        return f"Name: {self.name__}, Surname: {self.surname__}"

class StudentManagers:
    """
    A class for managing multiple student databases.
    
    The StudentManagers class provides functionality to create, manage, and operate on
    multiple student databases. It allows creating databases, switching between them,
    adding students, and displaying students in sorted order.
    
    Attributes
    ----------
    dbs : dict[str, list[Student]]
        A dictionary mapping database names to lists of Student objects
    current_db : str or None
        The name of the currently active database, or None if no database is selected
        
    Examples
    --------
    >>> sm = StudentManagers()
    >>> sm.create_db("class_2024")
    >>> sm.set_current_db("class_2024")
    >>> student = Student("John", "Doe")
    >>> sm.add_student(student)
    """
    
    def __init__(self):
        """
        Initialize a new StudentManagers instance.
        
        Creates an empty dictionary of databases and sets no current database.
        """
        # dict[str, list[Student]]
        self.dbs = {}
        self.current_db = None

    def create_db(self, db: str) -> None:
        """
        Create a new student database.
        
        Parameters
        ----------
        db : str
            The name of the database to create
            
        Raises
        ------
        ValueError
            If a database with the given name already exists
            
        Examples
        --------
        >>> sm = StudentManagers()
        >>> sm.create_db("class_2024")
        >>> "class_2024" in sm.dbs
        True
        """
        if db in self.dbs:
            raise ValueError(f"Database {db} already exists.")
        self.dbs[db] = []

    def set_current_db(self, db: str) -> None:
        """
        Set the current active database.
        
        Parameters
        ----------
        db : str
            The name of the database to set as current
            
        Raises
        ------
        ValueError
            If the specified database does not exist
            
        Examples
        --------
        >>> sm = StudentManagers()
        >>> sm.create_db("class_2024")
        >>> sm.set_current_db("class_2024")
        >>> sm.current_db
        'class_2024'
        """
        if db not in self.dbs:
            raise ValueError(f"Database {db} does not exist.")
        self.current_db = db
    
    def add_student(self, student: Student) -> None:
        """
        Add a student to the current database.
        
        Parameters
        ----------
        student : Student
            The Student instance to add to the current database
            
        Raises
        ------
        ValueError
            If no current database is set or if the student already exists in the database
            
        Examples
        --------
        >>> sm = StudentManagers()
        >>> sm.create_db("class_2024")
        >>> sm.set_current_db("class_2024")
        >>> student = Student("John", "Doe")
        >>> sm.add_student(student)
        >>> len(sm.dbs["class_2024"])
        1
        """
        if self.current_db is None:
            raise ValueError("No current database set.")
        if student in self.dbs[self.current_db]:
            raise ValueError(f"Student {student} already exists in database {self.current_db}.")
        self.dbs[self.current_db].append(student)
    
    def show_students(self) -> None:
        """
        Display all students in the current database in sorted order.
        
        Students are sorted by surname first, then by name. The output is printed
        to the console with each student on a separate line.
        
        Raises
        ------
        ValueError
            If no current database is set
            
        Examples
        --------
        >>> sm = StudentManagers()
        >>> sm.create_db("class_2024")
        >>> sm.set_current_db("class_2024")
        >>> sm.add_student(Student("John", "Doe"))
        >>> sm.add_student(Student("Jane", "Smith"))
        >>> sm.show_students()
        Name: John, Surname: Doe
        Name: Jane, Surname: Smith
        """
        if self.current_db is None:
            raise ValueError("No current database set.")
        students = sorted(self.dbs[self.current_db])
        for student in students:
            print(student)
