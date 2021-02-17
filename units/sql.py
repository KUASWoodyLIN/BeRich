import pymysql
import settings


class StockSQL:
    def __init__(self):
        host = settings.HOST
        port = settings.PORT
        user = settings.USER
        password = settings.PASSWORD
        dbname = settings.DBNAME
        charset = settings.CHARSET
        # 開啟資料庫連線
        self.db = pymysql.connect(host=host, port=port, user=user, password=password, db=dbname, charset=charset)

    def read_one_line(self, sql):
        """

        :param sql: "SELECT max(日期) FROM AI分析;"
        :return:
        """
        # 使用 cursor() 方法建立遊標物件 cursor
        with self.db.cursor() as cursor:
            # 使用 execute() 方法执行SQL查询
            cursor.execute(sql)
            # 使用 fetchone() 方法获取单条数据
            data = cursor.fetchone()
        return data

    def read_multi_line(self, sql):
        # 使用 cursor() 方法建立遊標物件 cursor
        with self.db.cursor() as cursor:
            # 使用 execute() 方法执行SQL查询
            cursor.execute(sql)
            # 使用 fetchall() 方法获取多条数据
            data = cursor.fetchall()
        return data

    def read_1723_stock_ids(self):
        """

        :return: type tuple(tuple(str)), (('1101'), ('2312'), ...)
        """
        select = "SELECT C.代號A \
                      FROM ( \
                            select A.代號A,B.代號 AS 代號B \
                            from (SELECT substr(代號名稱, 1, 4) as 代號A FROM base__股票代號) A \
                            LEFT JOIN base__下市股票 B ON A.代號A = B.代號 \
                            ) C \
                      WHERE C.代號B IS NULL \
                      ORDER BY C.代號A \
                      ;"
        # 使用 cursor() 方法建立遊標物件 cursor
        with self.db.cursor() as cursor:
            cursor.execute(select)
            result = cursor.fetchall()
        return result

    def read_1723_stocks_info(self):

        print()

    @staticmethod
    def read_171_stock_ids():
        """

        :return: type tuple(tuple(str)), (('1101'), ('2312'), ...)
        """
        import sqlite3
        import pandas as pd

        con = sqlite3.connect("dataset/Stock_Database.sqlite")
        data = pd.read_sql_query("SELECT * FROM Stock_No", con)
        stock_ids = data['StockNo']
        return stock_ids

    def read_stock_values(self, select):
        """
        :param select = "SELECT 日期, 開, 高, 低, 收, MA20, K9, D9, OSC, UB20, PB, BW, 成交量 \
                      FROM StockAll \
                      WHERE 代號 = '" + stock_id + "' \
                      ORDER BY 日期 DESC \
                      LIMIT 4000 \
                      ;"
        :return:
        """
        # 使用 cursor() 方法建立遊標物件 cursor
        with self.db.cursor() as cursor:
            cursor.execute(select)
            result = cursor.fetchall()
        return result

    def write_one_line(self, sql, ret):
        """

        :param sql:
        :param ret:
        :return:
        """
        # 使用 cursor() 方法建立遊標物件 cursor
        with self.db.cursor() as cursor:
            cursor.execute(sql, ret)
        self.db.commit()

    def write_multi_line(self, sql, ret):
        """

        :param sql: "INSERT INTO `" + 'AI分析' + "` VALUES (%s,%s,%s)"
        :param ret:
                ret = [
                        ('2020-12-22', '2330', '0.5458214'),
                        ('2020-12-22', '2330', '0.5458214'),
                        ('2020-12-22', '2330', '0.5458214')
                    ]
        :return:
        """
        # 使用 cursor() 方法建立遊標物件 cursor
        with self.db.cursor() as cursor:
            cursor.executemany(sql, ret)
        self.db.commit()


if __name__ == "__main__":
    stock_sql = StockSQL()
    # stock_sql.write_multi_line(
    #     "INSERT INTO `AI分析` VALUES (%s,%s,%s)",
    #     ret=[
    #         ('2020-12-22', '2330', '0.5458214'),
    #         ('2020-12-22', '2330', '0.5458214'),
    #         ('2020-12-22', '2330', '0.5458214')
    #     ]
    # )
    r = stock_sql.read_one_line("SELECT * FROM AI分析")
    r2 = stock_sql.read_multi_line("SELECT * FROM AI分析")
    print(r)
    print(r2)

    # 讀取Data
    # stock_ids = stock_sql.read_stock_ids()
    # v = stock_sql.read_stock_values(stock_ids[0][0])
    # print(v)
    print()
