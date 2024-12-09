from datetime import datetime
from functools import wraps
import os
import random
import time


def time_it(fn):
    @wraps(fn)
    def inner_func(*args, **kwargs):
        start_at = datetime.now()
        result = fn(*args, **kwargs)
        end_at = datetime.now()
        function_ran = end_at-start_at
        print(f"Function ran for {function_ran}")
        return result
    return inner_func

@time_it
def my_func_1(a, b):
    return a*b

# my_func(3, 4)

# Logging Function Calls
# Write a decorator log_call that logs the name of the function being called and its arguments.

def log_it(fn):
    @wraps(fn)
    def inner_func(*args, **kwargs):
        print("Function start")
        result = fn(*args, **kwargs)
        print("Func finished")
        return result
    return inner_func

@log_it
def some_func(*args):
    for arg in args:
        print(arg)

# some_func(1,2,3,4)

# Restrict Function Calls
# Create a decorator restrict_calls that raises an error if a function is called more than a specified number of times.

def restrict_calls(max_times):
    def decorator(fn):
        times = 0
        @wraps(fn)
        def inner_func(*args, **kwargs):
            nonlocal times
            print(times)
            if times >= max_times:
                raise ValueError("Too many times")
            result = fn(*args, **kwargs)
            times += 1
            return result
        return inner_func
    return decorator

@restrict_calls(max_times=3)
def my_func(a, b):
    return a*b

# my_func(1,2)
# my_func(1,3)
# my_func(1,3)
# my_func(1,3)
# my_func(1,3)



# Checking User Permissions
# Create a decorator check_permission that takes a role as an argument and only allows the function to execute if the user has that role.



# Retry Mechanism
# Write a decorator retry that retries a function if it raises an exception. It should accept a parameter max_retries to control the number of retries.

def retry(max_retries=3, delay=1):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    attempts +=1
                    print("Failed")
                    if attempts < max_retries:
                        time.sleep(delay)
            raise Exception("Func failed too many times")
        return wrapper
    return decorator


@retry(max_retries=1, delay=1)
def some_func():
    if random.choice([True, False]):
        print("Success")
        return "Res"
    else:
        raise ValueError("Err")

# some_func()



# Logging Function Execution with Arguments
# Write a decorator log_execution that logs the execution time, function name, and arguments passed to the decorated function.

# def log_fact():
#     def decorator(fn):
#         def wrapper(*args, **kwargs):
#             start_at = datetime.now()
#             result = fn(*args, **kwargs)
#             end_at = datetime.now()
#             timed = end_at - start_at
#             print(f"Func {fn.__name__} with args {args} and {kwargs} ran for {timed}")
#             return result
#         return wrapper
#     return decorator

def log_fact(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start_at = datetime.now()
        result = fn(*args, **kwargs)
        end_at = datetime.now()
        timed = end_at - start_at
        print(f"Func {fn.__name__} with args {args} and {kwargs} ran for {timed}")
        return result
    return wrapper

@log_fact
def one_func(a, b, c=5):
    print(a)
    return b

# one_func(1, 4, c=5)



# Customizable Decorator Factory
# Create a decorator factory repeat that takes an argument n and repeats the execution of the decorated function n times.

def repeat_dec(times):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for n in range(times):
                result = fn(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat_dec(4)
def rep_f():
    print("repeat")

# rep_f()


# Role-Based Access Control
# Build a more advanced role_required decorator that checks a user object for a specific role before allowing the decorated function to execute.

def role_required(role):
    def decorator(fn):
        @wraps(fn)
        def inner_func(*args, current_user=None, **kwargs):
            if current_user is None:
                raise ValueError("No user provided")
            if current_user.get("role") != role:
                raise PermissionError("Not permitted")
            return fn(*args, **kwargs)
        return inner_func
    return decorator





# current_user = {"username": "john_doe", "role": "admin"}

# @role_required(role="admin")
# def perform_admin_task():
#     print("Admin task performed!")


# Write a decorator validate_positive that ensures all numeric arguments passed to the decorated function are positive. Raise a ValueError if a negative number is passed.
def validate_positive(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        for n in args:
            if n < 0:
                raise ValueError("Negative")
        print("positive")
        for n in kwargs:
            if n<0:
                raise ValueError("Negative")
        return fn(*args, **kwargs)
    return wrapper

@validate_positive
def count_me(a, b):
    return a+b

# count_me(1, 5)
# count_me(-2, 5)
# print(count_me.__name__)

# Create a decorator to_uppercase that converts the output of a function to uppercase strings.

def to_upper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result.upper()
    return wrapper

@to_upper
def print_hello(name):
    return f"Hello, {name}"

# print(print_hello("A"))

# Implement a log_calls decorator that logs when a function is called, along with its arguments and return value.

def log_calls(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        time = datetime.now()
        result = fn(*args, *kwargs)
        print(f"Function is called at {time} with args {args} and kwargs {kwargs}. Result is {result}")
        return result
    return wrapper

@log_calls
def called_func(a, b):
    return a+b

# called_func(5, 8)


# Write a decorator enforce_types that enforces the types of arguments passed to a function. Use Python type hints.
import inspect
from typing import get_type_hints

def enforce_types(func):
    """
    A decorator that enforces the types of arguments passed to a function based on its type hints.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints of the function
        hints = get_type_hints(func)
        
        # Get argument names from the function's signature
        func_args = func.__code__.co_varnames[:func.__code__.co_argcount]
        
        # Map args and kwargs to parameter names
        bound_args = dict(zip(func_args, args))
        bound_args.update(kwargs)
        
        # Validate argument types
        for arg_name, arg_value in bound_args.items():
            if arg_name in hints:
                expected_type = hints[arg_name]
                if not isinstance(arg_value, expected_type):
                    raise TypeError(
                        f"Argument '{arg_name}' must be of type {expected_type}, got {type(arg_value)}"
                    )
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Validate return type if specified
        if 'return' in hints:
            expected_return_type = hints['return']
            if not isinstance(result, expected_return_type):
                raise TypeError(
                    f"Return value must be of type {expected_return_type}, got {type(result)}"
                )
        
        return result

    return wrapper
# def enforce_types(fn):
#     sig = inspect.signature(fn)
#     annotations = fn.__annotations__
#     @wraps(fn)
#     def wrapper(*args, **kwargs):
#         bound_args = sig.bind(*args, **kwargs)
#         bound_args.apply_defaults()
        
#         for name, value in bound_args.items():
#             if name in annotations:
#                 expected_type = annotations["name"]
#                 if not isinstance(value, expected_type):
#                     raise TypeError(f"arg {name} must be of type {expected_type}")
#         return fn(*args, **kwargs)
    # return wrapper

# @enforce_types
# def try_me(a: int, b: str):
#     print(a)
#     print(b)

# try_me("1", "a")

# Create a decorator measure_memory that prints the memory usage before and after a function call.

# import psutil

# def measure_memory(fn):
#     @wraps(fn)
#     def wrapper(*args, **kwargs):
#         process = psutil.Process(os.getpid())
#         memory_before = process.memory_info().rss / 1024**2
#         print(f"Memory usage before {fn.__name__}: {memory_before:.2f} MB")
#         result = fn(*args, **kwargs)
#         memory_after = process.memory_info().rss / 1024**2
#         print(f"Memory usage after {fn.__name__}: {memory_after:.2f} MB")
        
#         print(f"Memory used by {fn.__name__}: {memory_after - memory_before:.2f}")
#         return result
#     return wrapper




# import tracemalloc
# def measure_memory(fn):
#     @wraps(fn)
#     def wrapper(*args, **kwargs):
#         tracemalloc.start()
#         start_snapshot = tracemalloc.take_snapshot()
#         result = fn(*args, **kwargs)
#         end_snapshot = tracemalloc.take_snapshot()
#         tracemalloc.stop()
#         stats = end_snapshot.compare_to(start_snapshot, "lineno")
#         print(stats)
#         return result
#     return wrapper

# @measure_memory
# def func_to_measure(a, b):
#     return a+b

# func_to_measure(100, 200)


# Write a decorator simple_cache that caches the results of a function so that repeated calls with the same arguments don’t recalculate the result.
def simple_cache(fn):
    cache = {}
    @wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key in cache:
            print(f"Cache hit for {fn.__name__} with args={args} kwargs={kwargs}")
            return cache[key]
        print(f"Cache miss for {fn.__name__} with args={args} kwargs={kwargs}")
        result = fn(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper

@simple_cache
def get_count(a, b, name="default"):
    return a+b

# get_count(1, 4)
# get_count(1, 4)
# get_count(2, 5)
# get_count(1,4)
# get_count(1, 4, name="mymy")



# Combine multiple decorators (e.g., log_calls, validate_positive, to_uppercase) and analyze the order of execution.
@log_calls
@validate_positive
@simple_cache
def get_count(a, b, name="default"):
    return a+b

# get_count(1, 4)

# Modify the retry decorator to implement exponential backoff for delays between retries.


# Write a decorator require_permission that checks if the current user has a required permission level. Raise a PermissionError if the user lacks the required permission.



# def role_required(role):
#     def decorator(fn):
#         @wraps(fn)
#         def inner_func(*args, current_user=None, **kwargs):
#             if current_user is None:
#                 raise ValueError("No user provided")
#             if current_user.get("role") != role:
#                 raise PermissionError("Not permitted")
#             return fn(*args, **kwargs)
#         return inner_func
#     return decorator

def check_permission_dec(level):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, current_user=None, **kwargs):
            if current_user is None:
                raise PermissionError("No user")
            if level not in current_user.get("permissions"):
                raise PermissionError("No permission")
            return fn(*args, **kwargs)
        return wrapper
    return decorator
            
current_user = {"username": "john", "permissions": ["read", "write"]}

@check_permission_dec("delete")
def get_data(*args, current_user=None, **kwargs):
    print("got data")
    return

# get_data(current_user=current_user)




# Implement a decorator timeout that raises a TimeoutError if a function takes longer than a specified amount of time to execute.

import multiprocessing

class TimeoutError(Exception):
    """Custom exception to raise when a function times out."""
    pass

def timeout(seconds):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            result_queue = multiprocessing.Queue()
            
            def target(queue):
                try:
                    result = fn(*args, **kwargs)
                    queue.put(result)
                except Exception as e:
                    queue.put(e)
            
            process = multiprocessing.Process(target=target, args=(result_queue,))
            process.start()
            process.join(seconds)

            if process.is_alive():
                process.terminate()
                raise TimeoutError(f"Function '{fn.__name__}' timed out after {seconds} seconds")
            
            if not result_queue.empty():
                result = result_queue.get()
                if isinstance(result, Exception):
                    raise result
                return result
            raise TimeoutError(f"Function '{fn.__name__}' did not return any result")
        return wrapper
    return decorator

# Create a decorator singleton that ensures a class has only one instance.


# Write a decorator html_tag that wraps the output of a function in a specified HTML tag.
# @html_tag("b")


# Create a decorator call_counter that keeps track of how many times a function has been called and prints the count.


# Write a decorator validate_params that checks function arguments against specified regex patterns.
# @validate_params({"email": r"[^@]+@[^@]+\.[^@]+"})




