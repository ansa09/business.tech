#!/usr/bin/env python3
"""
Job Matcher Agent
Scores job descriptions against CVs using keyword matching
"""

import re
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter
from pypdf import PdfReader
from docx import Document


class JobMatcher:
    def __init__(self):
        """Initialize the Job Matcher"""
        # Common technical skills, tools, and keywords to look for
        self.common_keywords = {
            # Programming languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust',
            'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql',

            # Web technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django',
            'flask', 'fastapi', 'spring', 'asp.net', 'laravel',

            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'dynamodb', 'oracle', 'sql server',

            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github',
            'terraform', 'ansible', 'ci/cd', 'devops',

            # Data & AI
            'machine learning', 'deep learning', 'ai', 'nlp', 'computer vision',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'spark',

            # Methodologies
            'agile', 'scrum', 'kanban', 'waterfall', 'lean', 'tdd', 'bdd',

            # Soft skills
            'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
            'project management', 'stakeholder management'
        }

    def read_files_from_folder(self, folder_path: str) -> Dict[str, str]:
        """Read all files from a folder (supports PDF, DOCX, and text files)"""
        files_content = {}
        folder = Path(folder_path)

        if not folder.exists():
            print(f"Warning: Folder {folder_path} does not exist")
            return files_content

        for file_path in folder.glob("*"):
            if file_path.is_file():
                try:
                    content = self._extract_text_from_file(file_path)
                    if content:
                        files_content[file_path.name] = content
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        return files_content

    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from PDF, DOCX, or plain text files"""
        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            # Extract text from PDF
            reader = PdfReader(str(file_path))
            text = []
            for page in reader.pages:
                text.append(page.extract_text())
            return "\n".join(text)

        elif suffix == '.docx':
            # Extract text from Word document
            doc = Document(str(file_path))
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return "\n".join(text)

        else:
            # Assume it's a text file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text"""
        # Convert to lowercase for matching
        text_lower = text.lower()

        # Find all keywords present in the text
        found_keywords = set()

        for keyword in self.common_keywords:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(keyword.replace(' ', r'\s+')) + r'\b'
            if re.search(pattern, text_lower):
                found_keywords.add(keyword)

        # Also extract capitalized words (likely to be technologies/tools)
        # Look for words that are 2+ characters and contain letters
        words = re.findall(r'\b[A-Z][A-Za-z0-9+#\.]+\b', text)
        for word in words:
            if len(word) >= 2:
                found_keywords.add(word.lower())

        # Extract common patterns like "X years" for experience
        years_exp = re.findall(r'(\d+)\+?\s*years?', text_lower)
        if years_exp:
            found_keywords.add(f"{max(years_exp)} years experience")

        return found_keywords

    def score_job_against_cv(self, cv_content: str, job_description: str, job_name: str) -> Dict:
        """Score a job description against a CV using keyword matching"""

        # Extract keywords from CV and job description
        cv_keywords = self.extract_keywords(cv_content)
        job_keywords = self.extract_keywords(job_description)

        # Find matching and missing keywords
        matched_keywords = cv_keywords.intersection(job_keywords)
        missing_keywords = job_keywords - cv_keywords
        extra_keywords = cv_keywords - job_keywords

        # Calculate match score
        if len(job_keywords) > 0:
            score = int((len(matched_keywords) / len(job_keywords)) * 100)
        else:
            score = 0

        # Prepare strengths (matched keywords)
        strengths = sorted(list(matched_keywords))[:10]  # Top 10 matches
        if not strengths:
            strengths = ["No significant keyword matches found"]

        # Prepare gaps (missing keywords)
        gaps = sorted(list(missing_keywords))[:10]  # Top 10 gaps
        if not gaps:
            gaps = ["No significant gaps identified"]

        # Generate recommendation
        if score >= 80:
            recommendation = "Excellent match! Strong alignment between CV and job requirements."
        elif score >= 60:
            recommendation = "Good match. Most key requirements are met with some minor gaps."
        elif score >= 40:
            recommendation = "Moderate match. Consider highlighting transferable skills and addressing key gaps."
        elif score >= 20:
            recommendation = "Weak match. Significant skill gaps exist. May require upskilling or career pivot."
        else:
            recommendation = "Poor match. Very few requirements are met. Consider other opportunities."

        return {
            'job_name': job_name,
            'score': score,
            'strengths': strengths,
            'gaps': gaps,
            'recommendation': recommendation,
            'matched_count': len(matched_keywords),
            'total_required': len(job_keywords)
        }

    def process_all_jobs(self, cv_folder: str = "cv", jobs_folder: str = "jobs") -> List[Dict]:
        """Process all CVs and jobs, return ranked results"""

        # Read CV files (we'll use the first one, or combine if multiple)
        cv_files = self.read_files_from_folder(cv_folder)
        if not cv_files:
            raise ValueError(f"No CV files found in {cv_folder}/ folder")

        # Combine all CVs or use the first one
        if len(cv_files) == 1:
            cv_content = list(cv_files.values())[0]
            cv_name = list(cv_files.keys())[0]
            print(f"Using CV: {cv_name}\n")
        else:
            cv_content = "\n\n---\n\n".join(cv_files.values())
            print(f"Using {len(cv_files)} CV files combined\n")

        # Read job descriptions
        job_files = self.read_files_from_folder(jobs_folder)
        if not job_files:
            raise ValueError(f"No job description files found in {jobs_folder}/ folder")

        print(f"Found {len(job_files)} job descriptions to analyze\n")
        print("=" * 70)

        # Score each job
        results = []
        for i, (job_name, job_content) in enumerate(job_files.items(), 1):
            print(f"\nAnalyzing job {i}/{len(job_files)}: {job_name}...")
            result = self.score_job_against_cv(cv_content, job_content, job_name)
            results.append(result)
            print(f"Score: {result['score']}/100 ({result['matched_count']}/{result['total_required']} keywords matched)")

        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)

        return results

    def format_results(self, results: List[Dict]) -> str:
        """Format results as a readable string"""
        output = []
        output.append("=" * 70)
        output.append("JOB MATCHING RESULTS - RANKED SHORTLIST")
        output.append("(Based on Keyword Matching Analysis)")
        output.append("=" * 70)
        output.append("")

        for i, result in enumerate(results, 1):
            output.append(f"\n{'=' * 70}")
            output.append(f"RANK #{i}: {result['job_name']}")
            output.append(f"MATCH SCORE: {result['score']}/100")
            output.append(f"Keywords Matched: {result['matched_count']} out of {result['total_required']}")
            output.append(f"{'=' * 70}")

            output.append("\nMATCHED KEYWORDS (Your Strengths):")
            for strength in result['strengths']:
                output.append(f"  ✓ {strength}")

            output.append("\nMISSING KEYWORDS (Potential Gaps):")
            for gap in result['gaps']:
                output.append(f"  ✗ {gap}")

            output.append(f"\nRECOMMENDATION:")
            output.append(f"  {result['recommendation']}")
            output.append("")

        return "\n".join(output)

    def save_results(self, results: List[Dict], output_file: str = "results.txt"):
        """Save formatted results to a file"""
        formatted = self.format_results(results)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted)

        print(f"\n{'=' * 70}")
        print(f"Results saved to {output_file}")
        print(f"{'=' * 70}")

        return formatted


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("JOB MATCHER AGENT - Keyword-Based Analysis")
    print("=" * 70 + "\n")

    try:
        # Initialize matcher
        matcher = JobMatcher()

        # Process all jobs
        results = matcher.process_all_jobs(cv_folder="cv", jobs_folder="jobs")

        # Print and save results
        formatted_results = matcher.save_results(results, "results.txt")

        # Print summary to console
        print("\n" + "=" * 70)
        print("RANKED MATCHES:")
        print("=" * 70)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['job_name']}")
            print(f"   Score: {result['score']}/100 ({result['matched_count']}/{result['total_required']} keywords)")

        print("\nAnalysis complete!")

    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have:")
        print("  1. CV files in the cv/ folder")
        print("  2. Job description files in the jobs/ folder")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
