import json
import psycopg2
import os


def lambda_handler(event, context):
    try:
        # Parse incoming JSON body from API Gateway
        body = json.loads(event.get("body", "{}"))
        user_id = body.get("id")
        name = body.get("name")

        if not user_id or not name:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": 'Missing "id" or "name"'}),
            }

        conn = psycopg2.connect(
            host=os.environ["PGHOST"],
            database=os.environ["PGDATABASE"],
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
            port=os.environ.get("PGPORT", "5432"),
        )
        cur = conn.cursor()
        cur.execute("INSERT INTO users (id, name) VALUES (%s, %s)", (user_id, name))
        conn.commit()
        cur.close()
        conn.close()

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": "User inserted into PostgreSQL"}),
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
