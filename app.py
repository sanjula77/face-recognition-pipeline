def handler(request):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": '{"message": "Face Recognition API (Vercel)", "status": "healthy"}'
    }
