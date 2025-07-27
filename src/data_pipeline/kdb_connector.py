from qpython import qconnection
import pandas as pd

class KDBConnector:
    """
    Interface for connecting to kdb+ and executing queries.
    """

    def __init__(self, host: str, port: int, user: str = None, password: str = None):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.conn = None

    def connect(self):
        """
        Establish connection to kdb+ instance.
        """
        try:
            self.conn = qconnection.QConnection(host=self.host, port=self.port, username=self.user, password=self.password)
            self.conn.open()
        except Exception as e:
            print(f"Failed to connect to kdb+: {e}")
            self.conn = None

    def close(self):
        """
        Close the connection.
        """
        if self.conn:
            self.conn.close()

    def execute_query(self, query: str):
        """
        Execute a Q query and return the result.
        """
        if not self.conn or not self.conn.is_connected():
            self.connect()
        try:
            result = self.conn.sendSync(query)
            return result
        except Exception as e:
            print(f"Query failed: {e}")
            return None

    def fetch_dataframe(self, query: str) -> pd.DataFrame:
        """
        Execute a Q query and return the result as a Pandas DataFrame.
        """
        result = self.execute_query(query)
        if result is not None:
            try:
                df = pd.DataFrame(result)
                return df
            except Exception as e:
                print(f"Data conversion failed: {e}")
                return None
        return None