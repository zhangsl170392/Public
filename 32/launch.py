import fix_ssl
import sys
from streamlit.web.cli import main

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.port", "8888"]
    main()