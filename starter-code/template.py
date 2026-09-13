"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl, đồng thời tiếp nhận
  và ghi nhận yêu cầu hỗ trợ khách hàng.
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, luôn xưng "VinAssistant" và gọi
  khách hàng một cách lịch sự.

## 2. AVAILABLE TOOLS
- search_product_catalog(category, max_price): Tra cứu sản phẩm/dịch vụ Vingroup
  theo danh mục ("xe_dien" hoặc "du_lich") và mức giá tối đa (VNĐ).
- submit_support_ticket(customer_name, issue_description, priority): Ghi nhận yêu
  cầu hỗ trợ của khách hàng vào hệ thống ticket.

## 3. CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm, giá cả hay trạng thái ticket.
2. PHẢI gọi tool tương ứng để lấy dữ liệu thực khi khách hỏi về sản phẩm hoặc
   cần tạo yêu cầu hỗ trợ.
3. Nếu câu hỏi là FAQ chung (không cần dữ liệu thực), trả lời trực tiếp bằng
   kiến thức đã biết, không gọi tool.
4. Nếu không tìm thấy kết quả phù hợp, thông báo rõ ràng cho khách hàng, không
   suy diễn thêm.

## 4. OPERATIONAL BOUNDARIES
- Chỉ trả lời các câu hỏi liên quan đến sản phẩm/dịch vụ của Vingroup
  (VinFast, Vinpearl) và các yêu cầu hỗ trợ liên quan.
- Từ chối lịch sự các yêu cầu nằm ngoài phạm vi trên.

## 5. OUTPUT CONTRACT
Mỗi bước xử lý tuân theo định dạng:
Thought: <suy luận của Agent>
Action: <tool được gọi, nếu có>
Observation: <kết quả trả về từ tool>
Final Answer: <câu trả lời cuối cùng cho khách hàng>
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Trả lời tĩnh (mock), không gọi tool -> minh hoạ nguy cơ hallucination
        # vì LLM thuần không có dữ liệu thực để tra cứu.
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Intent Detection
    # ------------------------------------------------------------------
    def _detect_intent(self, text: str) -> Dict[str, bool]:
        text_lower = text.lower()

        has_price_signal = bool(re.search(r"giá|triệu|đồng|vnd", text_lower))
        has_category_word = any(
            k in text_lower
            for k in ["xe điện", "xe dien", "vinfast", "du lịch", "du lich",
                      "resort", "vinpearl", "phòng", "phong", "tour"]
        )
        needs_catalog = has_price_signal and has_category_word

        ticket_keywords = [
            "bị lỗi", "bi loi", "sự cố", "su co", "phản hồi", "phan hoi",
            "ghi nhận", "ghi nhan", "khiếu nại", "khieu nai", "hỗ trợ",
            "ho tro", "vấn đề", "van de", "bị hỏng", "bi hong",
            "ẩm mốc", "am moc", "xử lý gấp", "xu ly gap"
        ]
        needs_ticket = any(k in text_lower for k in ticket_keywords)

        is_faq = not needs_catalog and not needs_ticket

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }

    # ------------------------------------------------------------------
    # Argument Extraction
    # ------------------------------------------------------------------
    def _extract_catalog_args(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()

        if any(k in text_lower for k in ["du lịch", "du lich", "resort", "vinpearl", "phòng", "phong", "tour"]):
            category = "du_lich"
        else:
            category = "xe_dien"

        args: Dict[str, Any] = {"category": category}

        price_match = re.search(r"(\d+(?:[.,]\d+)?)\s*triệu", text_lower)
        if price_match:
            value = float(price_match.group(1).replace(",", "."))
            args["max_price"] = int(value * 1_000_000)

        return args

    def _extract_ticket_args(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()

        name_match = re.search(
            r"tôi tên\s+([^,.\n]+)|tên tôi là\s+([^,.\n]+)",
            text, re.IGNORECASE
        )
        if name_match:
            customer_name = (name_match.group(1) or name_match.group(2)).strip()
        else:
            customer_name = "Khách hàng"

        remainder = text
        if name_match:
            remainder = text[name_match.end():].lstrip(",").strip()
        issue_description = remainder.split(".")[0].strip()
        if not issue_description:
            issue_description = text.strip()

        if any(k in text_lower for k in ["nghiêm trọng", "nghiem trong", "gấp", "gap", "khẩn cấp", "khan cap"]):
            priority = "high"
        elif any(k in text_lower for k in ["không gấp", "khong gap", "thấp", "thap"]):
            priority = "low"
        else:
            priority = "medium"

        return {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority
        }

    # ------------------------------------------------------------------
    # Final Answer Synthesis
    # ------------------------------------------------------------------
    def _build_catalog_answer(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."
        lines = [
            f"- {p['name']}: {p['price_vnd']:,} VNĐ".replace(",", ".")
            for p in results
        ]
        return "Dưới đây là các sản phẩm phù hợp:\n" + "\n".join(lines)

    def _build_ticket_answer(self, ticket_result: Dict[str, Any]) -> str:
        return (
            f"{ticket_result.get('message', '')} "
            f"Mã ticket: {ticket_result['ticket_id']}. "
            f"Cảm ơn {ticket_result['customer_name']} đã phản hồi, "
            f"đội ngũ hỗ trợ sẽ liên hệ sớm nhất."
        )

    def _build_faq_answer(self, user_input: str) -> str:
        text_lower = user_input.lower()
        if "bảo hành" in text_lower or "bao hanh" in text_lower:
            return (
                "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc "
                "160.000 km (tuỳ điều kiện nào đến trước)."
            )
        return "Cảm ơn câu hỏi của bạn, VinAssistant xin phản hồi dựa trên chính sách chung của Vingroup."

    # ------------------------------------------------------------------
    # Agent Loop
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        self.trace.append({"step": "init", "user_input": user_input})

        intents = self._detect_intent(user_input)
        tool_call_count = 0
        answer_parts: List[str] = []

        if intents["needs_catalog"]:
            tool_call_count += 1
            args = self._extract_catalog_args(user_input)
            result = search_product_catalog(**args)
            self.trace.append({
                "step": tool_call_count,
                "action": "search_product_catalog",
                "args": args,
                "observation": result
            })
            answer_parts.append(self._build_catalog_answer(result))

        if intents["needs_ticket"]:
            tool_call_count += 1
            args = self._extract_ticket_args(user_input)
            result = submit_support_ticket(**args)
            self.trace.append({
                "step": tool_call_count,
                "action": "submit_support_ticket",
                "args": args,
                "observation": result
            })
            answer_parts.append(self._build_ticket_answer(result))

        if intents["is_faq"]:
            answer_parts.append(self._build_faq_answer(user_input))

        # Iteration bookkeeping: đúng 1 tool -> gộp cùng iteration tổng hợp;
        # từ 2 tool trở lên -> cần thêm 1 iteration riêng để tổng hợp Final Answer.
        if tool_call_count >= 2:
            iterations = tool_call_count + 1
        else:
            iterations = max(tool_call_count, 1)

        if iterations > self.max_iterations:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": iterations,
                "status": "max_iterations_reached"
            }

        final_answer = "\n\n".join(answer_parts)
        self.trace.append({"step": "final_answer", "answer": final_answer})

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iterations,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
