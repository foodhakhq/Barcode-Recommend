import multiprocessing

# Number of worker processes based on the number of CPU cores
workers = multiprocessing.cpu_count() * 2 + 1  # Adjust as needed

# Logging configuration
accesslog = '-'  # Logs to stdout
errorlog = '-'   # Logs to stderr
loglevel = 'info'  # Log level

# Timeout settings
timeout = 180  # or longer based on the longest expected processing time
graceful_timeout = 30  # Seconds to wait for workers to finish
keep_alive_timeout = 5  # Seconds to keep connections alive

# Maximum number of pending connections
backlog = 2048  # Adjust based on expected load

# Request handling
max_requests = 1000  # Max requests per worker before restart
max_requests_jitter = 100  # Random jitter to prevent simultaneous restarts
