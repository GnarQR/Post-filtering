import json
import re
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# 엑셀에서 허용되지 않는 제어 문자 필터링 정규식 (\t, \n, \r 제외)
ILLEGAL_CHARACTERS_RE = re.compile(
    r"[\000-\010]|[\013-\014]|[\016-\037]|[\x7f-\x9f]|[\ufffe-\uffff]"
)

def clean_text(val):
    """문자열에서 엑셀 비허용 제어 문자를 제거"""
    if isinstance(val, str):
        return ILLEGAL_CHARACTERS_RE.sub("", val)
    return val

def jsonl_to_excel(jsonl_file_path: str, excel_file_path: str):
    records = []
    
    # 1. JSONL 파일 읽기 (잘린 라인 예외 처리 포함)
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[경고] {line_no}번째 줄의 JSON 형식이 올바르지 않아 건너뜁니다.")

    summary_data = []
    detail_data = []

    for item in records:
        sample_id = clean_text(item.get("sample_id", ""))
        domain = clean_text(item.get("domain", ""))
        scope = clean_text(item.get("scope", ""))
        title = clean_text(item.get("title", ""))
        pages = ", ".join(map(str, item.get("pages", []))) if isinstance(item.get("pages"), list) else str(item.get("pages", ""))
        question = clean_text(item.get("question", ""))
        answer = clean_text(item.get("answer", ""))
        jev_suspect = item.get("jev_suspect", False)
        
        # 근거 청크 합치기 및 제어 문자 제거
        premises = item.get("premise_chunks", [])
        if isinstance(premises, list):
            premise_text = "\n\n--- [Chunk 구분] ---\n\n".join(premises)
        else:
            premise_text = str(premises)
        premise_text = clean_text(premise_text)
        
        # 문장별 jev 정보 확인
        jev_sents = item.get("jev_sentences", [])
        neu_count = sum(1 for s in jev_sents if s.get("jev_label") == "NEU")
        con_count = sum(1 for s in jev_sents if s.get("jev_label") == "CON")
        ent_count = sum(1 for s in jev_sents if s.get("jev_label") == "ENT")

        summary_data.append({
            "Sample ID": sample_id,
            "Domain": domain,
            "Scope": scope,
            "Document Title": title,
            "Pages": pages,
            "Hallucination Suspect": "SUSPECT" if jev_suspect else "CLEAN",
            "ENT Count": ent_count,
            "NEU Count": neu_count,
            "CON Count": con_count,
            "Question": question,
            "Generated Answer": answer,
            "Premise Chunks": premise_text
        })

        for sent_idx, sent_info in enumerate(jev_sents, 1):
            detail_data.append({
                "Sample ID": sample_id,
                "Domain": domain,
                "Sent #": sent_idx,
                "Sentence": clean_text(sent_info.get("sent", "")),
                "NLI Label": clean_text(sent_info.get("jev_label", "")),
                "P(Entailment)": sent_info.get("p_entailment", 0.0),
                "Sample Suspect": "SUSPECT" if jev_suspect else "CLEAN",
                "Related Question": question
            })

    df_summary = pd.DataFrame(summary_data)
    df_detail = pd.DataFrame(detail_data)

    # 2. 엑셀 파일 생성 및 스타일링
    with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="샘플별 요약", index=False)
        df_detail.to_excel(writer, sheet_name="문장별 NLI 검증 상세", index=False)
        
        wb = writer.book
        
        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        header_font = Font(name="맑은 고딕", size=10, bold=True, color="FFFFFF")
        regular_font = Font(name="맑은 고딕", size=9)
        suspect_fill = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid")
        con_fill = PatternFill(start_color="FFCDD2", end_color="FFCDD2", fill_type="solid")
        neu_fill = PatternFill(start_color="FFF9C4", end_color="FFF9C4", fill_type="solid")
        
        thin_border = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9")
        )

        for ws in wb.worksheets:
            # 헤더 스타일
            for col in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=col)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

            # 데이터 행 서식
            for row in range(2, ws.max_row + 1):
                for col in range(1, ws.max_column + 1):
                    cell = ws.cell(row=row, column=col)
                    cell.font = regular_font
                    cell.border = thin_border
                    
                    if ws.title == "샘플별 요약" and col in [10, 11, 12]:
                        cell.alignment = Alignment(vertical="top", wrap_text=True)
                    elif ws.title == "문장별 NLI 검증 상세" and col in [4, 8]:
                        cell.alignment = Alignment(vertical="top", wrap_text=True)
                    else:
                        cell.alignment = Alignment(vertical="center", horizontal="center" if col <= 5 else "left")

                    if cell.value == "SUSPECT":
                        cell.fill = suspect_fill
                        cell.font = Font(name="맑은 고딕", size=9, bold=True, color="C62828")
                    elif cell.value == "CON":
                        cell.fill = con_fill
                        cell.font = Font(name="맑은 고딕", size=9, bold=True, color="B71C1C")
                    elif cell.value == "NEU":
                        cell.fill = neu_fill

            # 컬럼 기본 너비 세팅
            for col in ws.columns:
                col_letter = get_column_letter(col[0].column)
                if ws.title == "샘플별 요약":
                    widths = {1: 28, 2: 12, 3: 10, 4: 25, 5: 12, 6: 15, 7: 10, 8: 10, 9: 10, 10: 40, 11: 45, 12: 50}
                    ws.column_dimensions[col_letter].width = widths.get(col[0].column, 20)
                else:
                    widths = {1: 28, 2: 12, 3: 8, 4: 55, 5: 12, 6: 15, 7: 15, 8: 45}
                    ws.column_dimensions[col_letter].width = widths.get(col[0].column, 20)

    print(f"변환 완료: '{excel_file_path}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    jsonl_to_excel("jev_scouted_qna.jsonl", "jev_scouted_qna_analysis.xlsx")