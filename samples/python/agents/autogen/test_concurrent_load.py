import asyncio
import aiohttp
import time
import uuid
import json
import argparse
import statistics
from datetime import datetime
import csv
import os

"""
This script performs load testing on the A2A server by sending multiple concurrent requests.
It measures response times, success rates, and other metrics to evaluate server performance under load.

Usage:
    python test_concurrent_load.py --url http://localhost:10002 --concurrency 10 --requests 100 --output results.csv
"""

class LoadTester:
    def __init__(self, server_url, concurrency=10, total_requests=100, output_file=None):
        self.server_url = server_url
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.output_file = output_file
        self.results = []
        self.agent_card = None
        self.session = None
        
    async def initialize(self):
        """Initialize the load tester by fetching the agent card."""
        async with aiohttp.ClientSession() as session:
            self.session = session
            agent_card_url = f"{self.server_url}/.well-known/agent.json"
            async with session.get(agent_card_url) as response:
                if response.status == 200:
                    self.agent_card = await response.json()
                    print(f"Successfully fetched agent card: {self.agent_card['name']}")
                else:
                    raise Exception(f"Failed to fetch agent card. Status: {response.status}")
    
    async def send_request(self, request_number):
        """Send a single request to the server and record metrics."""
        if not self.agent_card:
            raise Exception("Agent card not initialized. Call initialize() first.")
            
        # Generate unique IDs for the request
        task_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Create request payload
        payload = {
            "id": request_number,
            "method": "sendTaskStreaming",
            "params": {
                "id": task_id,
                "sessionId": session_id,
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": f"Load test request #{request_number} at {datetime.now().isoformat()}"
                        }
                    ]
                },
                "acceptedOutputModes": ["text", "text/plain"]
            }
        }
        
        start_time = time.time()
        success = False
        error = None
        response_count = 0
        
        try:
            # Send the request
            async with self.session.post(self.server_url, json=payload) as response:
                if response.status == 200:
                    # Process the SSE stream
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data:'):
                            response_count += 1
                            data = json.loads(line[5:])
                            if data.get('result', {}).get('final', False):
                                success = True
                                break
                else:
                    error = f"HTTP {response.status}: {await response.text()}"
        except Exception as e:
            error = str(e)
            
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            "request_id": request_number,
            "task_id": task_id,
            "session_id": session_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "success": success,
            "error": error,
            "response_count": response_count
        }
        
        self.results.append(result)
        return result
    
    async def run_test(self):
        """Run the load test with the specified concurrency and number of requests."""
        print(f"Starting load test with {self.concurrency} concurrent requests, {self.total_requests} total")
        
        # Initialize first to get agent card
        await self.initialize()
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def limited_request(i):
            async with semaphore:
                result = await self.send_request(i)
                print(f"Request {i}: {'✅' if result['success'] else '❌'} {result['duration']:.2f}s")
                return result
                
        # Run all requests with limited concurrency
        tasks = [limited_request(i) for i in range(1, self.total_requests + 1)]
        await asyncio.gather(*tasks)
        
        # Calculate and print metrics
        self.print_metrics()
        
        # Save results to CSV if output file specified
        if self.output_file:
            self.save_results()
    
    def print_metrics(self):
        """Calculate and print test metrics."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful
        
        durations = [r["duration"] for r in self.results if r["success"]]
        
        if durations:
            avg_duration = statistics.mean(durations)
            median_duration = statistics.median(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            p95_duration = sorted(durations)[int(len(durations) * 0.95)]
        else:
            avg_duration = median_duration = min_duration = max_duration = p95_duration = 0
        
        print("\n=== Load Test Results ===")
        print(f"Total requests: {total}")
        print(f"Successful: {successful} ({(successful/total)*100:.2f}%)")
        print(f"Failed: {failed} ({(failed/total)*100:.2f}%)")
        print(f"Average response time: {avg_duration:.2f}s")
        print(f"Median response time: {median_duration:.2f}s")
        print(f"Min response time: {min_duration:.2f}s")
        print(f"Max response time: {max_duration:.2f}s")
        print(f"95th percentile: {p95_duration:.2f}s")
        
        # Print errors grouped by error message
        if failed > 0:
            error_counts = {}
            for r in self.results:
                if not r["success"] and r["error"]:
                    error_counts[r["error"]] = error_counts.get(r["error"], 0) + 1
                    
            print("\nError breakdown:")
            for error, count in error_counts.items():
                print(f"  {count} requests: {error}")
    
    def save_results(self):
        """Save detailed results to a CSV file."""
        with open(self.output_file, 'w', newline='') as csvfile:
            fieldnames = [
                "request_id", "task_id", "session_id", "start_time", "end_time",
                "duration", "success", "error", "response_count"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
                
        print(f"Detailed results saved to {self.output_file}")


async def main():
    parser = argparse.ArgumentParser(description='Load testing for A2A Server')
    parser.add_argument('--url', required=True, help='URL of the A2A server')
    parser.add_argument('--concurrency', type=int, default=10, help='Number of concurrent requests')
    parser.add_argument('--requests', type=int, default=100, help='Total number of requests to send')
    parser.add_argument('--output', help='Output CSV file for detailed results')
    
    args = parser.parse_args()
    
    tester = LoadTester(
        server_url=args.url,
        concurrency=args.concurrency,
        total_requests=args.requests,
        output_file=args.output
    )
    
    await tester.run_test()


if __name__ == "__main__":
    asyncio.run(main()) 