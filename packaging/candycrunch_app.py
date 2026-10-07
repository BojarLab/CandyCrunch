"""Entry point of the bundled CandyCrunch app (built with packaging/candycrunch.spec)"""
import multiprocessing
if __name__ == '__main__':
    # Turns a worker process start into the worker instead of a second window
    multiprocessing.freeze_support()
    from candycrunch.gui import main
    main()
