import subprocess
import glob
import os
import sys
import time

def run_metta_tests():
    # 1. Dynamically find the paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.abspath(os.path.join(script_dir, "..", "tests"))
    
    # 2. Look for files starting with 'test' and ending in '.metta'
    search_pattern = os.path.join(tests_dir, "test*.metta")
    test_files = glob.glob(search_pattern)
    
    if not test_files:
        print(f"[INFO] No test files matching 'test*.metta' found in: {tests_dir}")
        sys.exit(1)
        
    passed_tests = len(test_files)
    failed_tests = 0

    all_tests_passed = True
    print(f"[INFO] Found {len(test_files)} test file(s) in {tests_dir}.")
    print("[INFO] Starting test run...\n")
    print("-" * 40)

    # Start the master timer for the entire suite
    suite_start_time = time.time()

    for file in test_files:
        filename = os.path.basename(file)
        print(f"Executing: {filename}...")
        
        # Start the timer for this specific file
        file_start_time = time.time()
        
        # 3. Execute the MeTTa file
        try:
            result = subprocess.run(
                ['../../../../other_tasks/hypron_tut/PeTTa/run.sh', file], 
                capture_output=True, 
                text=True,
                encoding='utf-8' # Ensures Python correctly reads the emoji characters
            )
            
            # Stop the individual timer and calculate duration
            file_end_time = time.time()
            file_duration = file_end_time - file_start_time
            
            print(f"result: {result.stdout}")
            # 4. Check for the specific '❌' fail marker in the output
            # We also keep the return code check in case the script crashes completely
            failed = (
                result.returncode != 0 or 
                "❌" in result.stdout or 
                "❌" in result.stderr
            )
            
            # Ensure it actually ran and passed a test
            passed = not failed and ("✅" in result.stdout or "✅" in result.stderr)

            if failed or not passed:
                failed_tests += 1
                passed_tests -= 1
                # Added time to the fail message
                print(f"❌ FAIL {filename} ({file_duration:.2f}s)\n")
                all_tests_passed = False
            else:
                # Added time to the pass message
                print(f"✅ PASS {filename} ({file_duration:.2f}s)\n")
                
        except FileNotFoundError:
            print("[ERROR] The 'petta' command was not found.")
            print("[INFO] Make sure the Petta/MeTTa interpreter is installed and added to your system's PATH.")
            sys.exit(1)

    # Stop the master timer
    suite_end_time = time.time()
    total_duration = suite_end_time - suite_start_time

    # 5. Final summary
    print("-" * 40)
    print(f"Total Time: {total_duration:.2f} seconds")
    if all_tests_passed:
        print(f"[SUCCESS] All logical proofs passed.")
        sys.exit(0)
    else:
        print(f"[FAILURE] {failed_tests} proofs failed out of {len(test_files)}.")
        sys.exit(1)

if __name__ == "__main__":
    run_metta_tests()
