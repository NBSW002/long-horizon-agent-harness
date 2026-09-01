def blocking_job() -> None:
    print("Starting blocking job")
    start_time = time.time()
    time.sleep(5)
    end_time = time.time()
    during = end_time - start_time
    print(f"Blocking job finished in {during:.2f} seconds")