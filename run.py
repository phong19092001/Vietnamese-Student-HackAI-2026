import json
import re
import os
import csv
import time
from google import genai

# API
API_KEY = open("key.txt", "r", encoding="utf-8").read().strip()
client = genai.Client(api_key=API_KEY)

def call_llm(prompt):
    while True:
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )
            return response.text

        except Exception as e:
            print("Lỗi:", e)
            print("Đợi 70 giây...")
            time.sleep(70)

# Prompt Solver
PROMPT = """
Bạn là chuyên gia phân tích câu hỏi và là chuyên gia trong nhiều lĩnh vực.

Nhiệm vụ:
1. Đọc câu hỏi.
2. Phân tích câu hỏi.
3. Phân tích từng đáp án.
4. Loại các đáp án sai.
5. Chọn đáp án đúng nhất.
6. Bắt buộc xem xét tất cả các đáp án trước khi đưa ra kết luận.
7. Với mỗi đáp án sai, xác định ngắn gọn lý do bị loại.

Lưu ý:
- Số lượng đáp án có thể từ 2 đến 10.
- Ký hiệu đáp án tương ứng:
A, B, C, D, E, F, G, H, I, J, K
- Chỉ được chọn MỘT đáp án cuối cùng.
- Không được trả về nhiều đáp án.

Trả về đúng định dạng:

ANSWER: C

REASON:
Giải thích ngắn gọn lý do chọn đáp án.
"""

# Đọc dữ liệu
import os

if os.path.exists("/data/private_test.json"):
    input_file = "/data/private_test.json"

elif os.path.exists("/data/public_test.json"):
    input_file = "/data/public_test.json"

else:
    input_file = "public_test.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Lấy đúng 1 câu đầu tiên
same_count = 0
diff_count = 0
results = []
for question in data[:230]:

    # Ghép lựa chọn
    choices_text = ""

    for i, choice in enumerate(question["choices"]):
        letter = chr(ord("A") + i)
        choices_text += f"{letter}. {choice}\n"

    # Prompt hoàn chỉnh
    full_prompt = f"""
    {PROMPT}

    CÂU HỎI:
    {question["question"]}

    LỰA CHỌN:
    {choices_text}
    """

    # Gọi Gemini
    solver_text = call_llm(full_prompt)
    response = type("obj", (), {"text": solver_text})


    # Parse ANSWER
    answer_match = re.search(
        r"ANSWER:\s*([A-K])",
        response.text,
        re.IGNORECASE
    )
 
    if answer_match:
        answer = answer_match.group(1)
        reason_match = re.search(
            r"REASON:\s*(.*)",
            response.text,
            re.DOTALL
        )

        if reason_match:
            solver_reason = reason_match.group(1)[:500]
        else:
            solver_reason = ""
        reason = response.text
    else:
        answer = "A"

    
    critic_prompt = f"""
    CÂU HỎI:

    {question["question"]}
  
    LỰA CHỌN:

    {choices_text}

    SOLVER ĐÃ CHỌN:

    {answer}

    PHÂN TÍCH CỦA SOLVER:

    {reason}

    Nhiệm vụ:

    Bạn là Prosecutor.

    Mục tiêu của bạn KHÔNG phải tìm cách đổi đáp án.

    Mục tiêu của bạn là tìm lỗi thực sự trong lập luận của Solver.

    QUY TẮC:

    1. Mặc định tin Solver.

    2. Chỉ được đổi đáp án Solver nếu:
    - Critic đưa ra bằng chứng cụ thể.
    - Judge tự kiểm tra lại đề và xác nhận bằng chứng đó đúng.

    Nếu không xác nhận được:
    giữ nguyên đáp án Solver.

    3. Bằng chứng hợp lệ gồm:
    - Chi tiết trong đề bài.
    - Chi tiết trong đáp án.
    - Công thức.
    - Phép tính.
    - Mâu thuẫn logic rõ ràng.

    4. Nếu không tìm thấy bằng chứng cụ thể:

    FINAL: giữ nguyên đáp án của Solver.

    REASON:
    Không tìm thấy lỗi đủ mạnh để bác bỏ Solver.

    5. Không được đổi đáp án chỉ vì:
    - Critic nghe có vẻ hợp lý.
    - Critic viết dài hơn.
    - Critic tự tin hơn.

    6. Không được suy đoán.

    Trả về đúng định dạng:

    ANSWER: <A-K>

    REASON:
    Nêu bằng chứng cụ thể.

    """
    critic_text = call_llm(critic_prompt)
    critic_response = type("obj", (), {"text": critic_text})
    critic_reason = critic_response.text[:500]
    critic_match = re.search(
    r"ANSWER:\s*([A-K])",
    critic_response.text,
    re.IGNORECASE
    )

    if critic_match:
        critic_answer = critic_match.group(1)
    else:
        critic_answer = answer

    valid_letters = [
        chr(ord("A") + i)
        for i in range(len(question["choices"]))
    ]
 

    if critic_answer not in valid_letters:
        print(
            "INVALID CRITIC:",
            critic_answer,
            "-> dùng đáp án Solver"
        )
        critic_answer = answer
        
    if answer == critic_answer:
        same_count += 1
        final_answer = answer

    else:
        diff_count += 1

        judge_prompt = f"""
        Bạn là Judge.

        Mục tiêu của bạn KHÔNG phải chọn phe.

        Bạn phải tự đọc lại:
         - câu hỏi
         - tất cả đáp án
         - lập luận của Solver
         - lập luận của Critic

        QUY TẮC:

        1. Mặc định tin Solver.

        2. Chỉ được đổi đáp án nếu Critic đưa ra bằng chứng cụ thể cho thấy Solver sai.

        3. Bằng chứng hợp lệ gồm:
        - dữ kiện trong đề
        - dữ kiện trong đáp án
        - công thức
        - phép tính
        - mâu thuẫn logic rõ ràng

        4. Không được đổi đáp án chỉ vì:
        - Critic nghe có vẻ hợp lý
        - Critic viết dài hơn
        - Critic tự tin hơn

        5. Nếu còn nghi ngờ:
        giữ nguyên đáp án của Solver.

        ========================

        CÂU HỎI:

        {question["question"]}

        LỰA CHỌN:

        {choices_text}

        ========================

        SOLVER
 
        ĐÁP ÁN:
        {answer}

        LẬP LUẬN:
        {solver_reason}
        ========================
 
        CRITIC

        ĐÁP ÁN:
        {critic_answer}

        LẬP LUẬN:
        {critic_reason}

        ========================

        Trả về đúng định dạng:

        FINAL: <A-K>

        DECISION:
        KEEP_SOLVER
        hoặc
        OVERTURN_SOLVER

        REASON:
        Giải thích ngắn gọn.
        """
        judge_text = call_llm(judge_prompt)
        judge_response = type("obj", (), {"text": judge_text})
        judge_match = re.search(
            r"FINAL:\s*([A-K])",
            judge_response.text,
            re.IGNORECASE
        )

        if judge_match:
            final_answer = judge_match.group(1)
        else:
            final_answer = answer
        if final_answer not in valid_letters:
            print(
                "INVALID JUDGE:",
                final_answer,
                "-> dùng đáp án Solver"
            )
            final_answer = answer
    # luôn chạy
    results.append({
        "qid": question["qid"],
        "answer": final_answer
    })

    print(f"{question['qid']} -> {final_answer}")

print("Same:", same_count)
print("Different:", diff_count)

if os.path.exists("/output"):
    output_file = "/output/pred.csv"
else:
    output_file = "result.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["qid", "answer"])

    for row in results:
        writer.writerow([
            row["qid"],
            row["answer"]
        ])
    for row in results:
        writer.writerow([
            row["qid"],
            row["answer"]
        ])

print("Saved:", len(results), "answers")
print("File: /output/pred.csv")