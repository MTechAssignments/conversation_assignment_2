import zipfile
import os
from bs4 import BeautifulSoup
import re


"""## 1. Data Collection & Preprocessing

Downloaded Financial Statement of GE Healthcare From United States Securities and Exchange Commission:

Click [here](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001932393&type=10-Q&dateb=&owner=include&count=40&search_text=) for link to source of data.

### 1.1 Extract the data and convert them to plain text. (Source data is html files)
"""

class FinancialReportProcessor:
    def __init__(self, zip_file_path, extracted_dir_path, plain_text_dir_path):
        self.zip_file_path = zip_file_path
        self.extracted_dir_path = extracted_dir_path
        self.plain_text_dir_path = plain_text_dir_path
        self.plain_text_data = []
        self.cleaned_text_data = []
        self.segmented_financial_statements = []
        self.financial_data = []
        self.generated_questions = []

    def extract_and_convert_html_to_text(self):
        os.makedirs(self.extracted_dir_path, exist_ok=True)
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.extracted_dir_path)
        print(f"Extracted {self.zip_file_path} to {self.extracted_dir_path}")

        os.makedirs(self.plain_text_dir_path, exist_ok=True)
        html_files = []
        for root, _, files in os.walk(self.extracted_dir_path):
            for file in files:
                if file.endswith(".html") or file.endswith(".htm"):
                    html_files.append(os.path.join(root, file))
        print(f"Found {len(html_files)} HTML files.")

        for html_file_path in html_files:
            try:
                try:
                    with open(html_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                except UnicodeDecodeError:
                    with open(html_file_path, 'r', encoding='latin-1') as f:
                        html_content = f.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                plain_text = soup.get_text(separator='\n')
                relative_path = os.path.relpath(html_file_path, self.extracted_dir_path)
                plain_text_file_path = os.path.join(self.plain_text_dir_path, relative_path + ".txt")
                os.makedirs(os.path.dirname(plain_text_file_path), exist_ok=True)
                with open(plain_text_file_path, 'w', encoding='utf-8') as f:
                    f.write(plain_text)
                print(f"Converted {html_file_path} to plain text and saved to {plain_text_file_path}")
            except Exception as e:
                print(f"Error processing {html_file_path}: {e}")
        print("Finished converting HTML files to plain text.")
        return len(html_files), self.plain_text_dir_path

    def load_plain_text_files(self):
        self.plain_text_data = []
        for root, _, files in os.walk(self.plain_text_dir_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self.plain_text_data.append(f.read())
                        print(f"Loaded {file_path}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        print(f"Loaded {len(self.plain_text_data)} plain text files.")
        return self.plain_text_data

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\[\s*\d+\s*\]', '', text)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Exhibit\s+\d+\.\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text

    def clean_all_texts(self):
        self.cleaned_text_data = [self.clean_text(text) for text in self.plain_text_data]
        return self.cleaned_text_data

    def segment_reports(self):
        financial_segment_patterns = {
            "Statements of Operations / Income": [
                r"CONSOLIDATED STATEMENTS OF OPERATIONS\s*\n(.*?)(?=\n(?:Statements of Financial Position|Statements of Comprehensive Income|Statements of Changes in Equity|balance sheet|cash flows)|\Z)",
                r"Statements of Income\s*\n(.*?)(?=\n(?:Statements of Financial Position|Statements of Comprehensive Income|Statements of Changes in Equity|balance sheet|cash flows)|\Z)",
            ],
            "Statements of Financial Position / Balance Sheet": [
                r"Statements of Financial Position\s*\n(.*?)(?=\n(?:Statements of Comprehensive Income|Statements of Changes in Equity|balance sheet|cash flows)|\Z)",
                r"balance sheet\s*\n(.*?)(?=\n(?:Statements of Comprehensive Income|Statements of Changes in Equity|cash flows)|\Z)",
            ],
            "Statements of Cash Flows": [
                r"cash flows\s*\n(.*?)(?=\Z)",
            ]
        }
        self.segmented_financial_statements = []
        for doc_text in self.cleaned_text_data:
            doc_segments = {}
            remaining_text = doc_text
            for segment_name, patterns in financial_segment_patterns.items():
                found_segment = False
                for pattern in patterns:
                    match = re.search(pattern, remaining_text, re.DOTALL | re.IGNORECASE)
                    if match:
                        doc_segments[segment_name] = match.group(1).strip()
                        remaining_text = remaining_text[match.end():]
                        found_segment = True
                        break
                if not found_segment:
                    doc_segments[segment_name] = "Segment not found."
            self.segmented_financial_statements.append(doc_segments)
        print(f"Segmented {len(self.segmented_financial_statements)} documents into financial statements.")
        return self.segmented_financial_statements

    def extract_financial_data(self):
        line_item_pattern = re.compile(
            r"^(.*?)\s+"
            r"\$\s*([\d,]+)\s+"
            r"\$\s*([\d,]+)\s+"
            r"\$\s*([\d,]+)\s+",
            re.MULTILINE
        )
        self.financial_data = []
        for i, doc_segments in enumerate(self.segmented_financial_statements):
            for segment_name, content in doc_segments.items():
                if content != "Segment not found.":
                    matches = line_item_pattern.finditer(content)
                    for match in matches:
                        line_item = match.group(1).strip()
                        value_2024 = match.group(2).replace(',', '')
                        value_2023 = match.group(3).replace(',', '')
                        value_2022 = match.group(4).replace(',', '')
                        self.financial_data.append({
                            "document": i + 1,
                            "segment": segment_name,
                            "line_item": line_item,
                            "2024": int(value_2024),
                            "2023": int(value_2023),
                            "2022": int(value_2022)
                        })
        for item in self.financial_data[:5]:
            print(item)
        return self.financial_data

    def generate_questions(self, max_questions=50):
        self.generated_questions = []
        for item in self.financial_data:
            line_item = item["line_item"]
            value_2024 = item["2024"]
            value_2023 = item["2023"]
            value_2022 = item["2022"]
            segment = item["segment"]
            self.generated_questions.append({
                "based_on_data_item": item,
                "question": f"What was the value of '{line_item}' in {2024} according to the {segment}?",
            })
            if len(self.generated_questions) == max_questions:
                break
            self.generated_questions.append({
                "based_on_data_item": item,
                "question": f"Find the value for '{line_item}' in {2023} from the {segment}.",
            })
            if len(self.generated_questions) == max_questions:
                break
            self.generated_questions.append({
                "based_on_data_item": item,
                "question": f"Could you provide the figure for '{line_item}' in {2022} as reported in the {segment}?",
            })
            if len(self.generated_questions) == max_questions:
                break
            self.generated_questions.append({
                "based_on_data_item": item,
                "question": f"How much did the '{line_item}' change from {2023} to {2024} based on the {segment}?",
            })
            if len(self.generated_questions) == max_questions:
                break
            self.generated_questions.append({
                "based_on_data_item": item,
                "question": f"What was the difference in '{line_item}' between {2022} and {2023} according to the {segment}?",
            })
            if len(self.generated_questions) == max_questions:
                break
            if value_2024 is not None and value_2023 is not None and value_2022 is not None:
                self.generated_questions.append({
                    "based_on_data_item": item,
                    "question": f"What were the values for '{line_item}' for the years {2024}, {2023}, and {2022} in the {segment}?",
                })
            if len(self.generated_questions) == max_questions:
                break
        for q in self.generated_questions[:10]:
            print(q)
        print(f"\nGenerated {len(self.generated_questions)} questions.")
        return self.generated_questions

    def generate_answers(self):
        for q in self.generated_questions:
            item = q["based_on_data_item"]
            line_item = item["line_item"]
            value_2024 = item["2024"]
            value_2023 = item["2023"]
            value_2022 = item["2022"]
            segment = item["segment"]
            question_text = q["question"]
            answer = ""
            if f"in {2024}" in question_text and f"{2023}, and {2022}" not in question_text:
                answer = f"The value of '{line_item}' in 2024 was {value_2024} millions of dollars."
            elif f"in {2023}" in question_text and f"{2024}, and {2022}" not in question_text:
                answer = f"The value of '{line_item}' in 2023 was {value_2023} millions of dollars."
            elif f"in {2022}" in question_text and f"{2024}, and {2023}" not in question_text:
                answer = f"The value of '{line_item}' in 2022 was {value_2022} millions of dollars."
            elif f"change from {2023} to {2024}" in question_text:
                change = value_2024 - value_2023
                answer = f"The change in '{line_item}' from 2023 to 2024 was {change} millions of dollars."
            elif f"difference in '{line_item}' between {2022} and {2023}" in question_text:
                difference = value_2023 - value_2022
                answer = f"The difference in '{line_item}' between 2022 and 2023 was {difference} millions of dollars."
            elif f"for the years {2024}, {2023}, and {2022}" in question_text:
                answer = f"The values for '{line_item}' for the years 2024, 2023, and 2022 were {value_2024}, {value_2023}, and {value_2022} millions of dollars, respectively."
            else:
                answer = "Could not determine the specific answer based on the question format."
            q['answer'] = answer
        print("First 10 Generated Q/A Pairs:")
        for q in self.generated_questions[:10]:
            print(f"Question: {q['question']}")
            print(f"Answer: {q['answer']}")
            print("-" * 20)
        print(f"\nTotal Q/A pairs generated: {len(self.generated_questions)}")
        return self.generated_questions

