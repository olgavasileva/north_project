# AWS SAM Sample App

This is a sample AWS SAM app with:

- A TypeScript Lambda function that writes to PostgreSQL.
- A Python Lambda function that writes to PostgresSQL.

## Prerequisites

- Docker (for local PostgreSQL testing)
- AWS CLI
- AWS SAM CLI
- Node.js (v18+)
- Python 3.11

## Setup Instructions

````bash
# Install TypeScript Lambda dependencies
cd ts-lambda
npm install
cd ..

# Compile TypeScript
npx tsc -p ts-lambda

# Build the SAM app
sam build



## PostgreSQL Setup (Local Docker)

```bash
docker run --name postgres -e POSTGRES_PASSWORD=example -e POSTGRES_DB=users -p 5432:5432 -d postgres
````

Once running, initialize the table:

```bash
docker exec -i postgres psql -U postgres -d users < db/init.sql
```

windows

```bash
Get-Content db/init.sql | docker exec -i postgres psql -U postgres -d users

```

## start local API

```bash
sam local start-api
```

### payload example

```json
{
  "name": "BOOM",
  "id": "739e9599-ca49-40dd-8da2-0acf7506b152"
}
```

#### endpoints

- POST /users
- POST /user-python
