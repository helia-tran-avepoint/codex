from genericpath import isfile
import os
import re
from pygount import LanguageSummary, SourceAnalysis, ProjectSummary
import argparse
from datetime import datetime
from git import Commit, Repo
from typing import List, Dict, Any, Tuple
import json
from git.exc import GitCommandError
import git
import pandas as pd

def analyze_source_file(file_path, group="CSharp", encoding="utf-8"):
    """
    分析单个 C# 文件的基础信息：
      - 利用 SourceAnalysis.from_file 统计代码行数、注释行数、空行数、字符串行数
      - 直接读取文件内容统计总行数和通过正则匹配统计类数量
    """
    try:
        # 使用新的工厂方法对文件进行分析
        analysis = SourceAnalysis.from_file(file_path, group, encoding=encoding)
    except Exception as e:
        print(f"分析文件 {file_path} 时出错：{e}")
        return None

    return analysis


def analyze_directory(directory, group="CSharp", encoding="utf-8"):
    """
    新增方法：利用 ProjectSummary 对指定目录下所有 C# 文件进行整体统计。
    遍历目录，调用 analyze_cs_file 得到每个文件的 SourceAnalysis 对象，
    并将其加入 ProjectSummary 中，最后返回整体统计数据。
    """
    analyzer = CodeCommitAnalyzer(directory)
    project_summary = ProjectSummary()
    results = []
    blamResults = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not should_exclude(d, analyzer)]
        for file in files:
                file_path = os.path.join(root, file)
                analysis = analyze_source_file(file_path, group, encoding)
                if analysis:
                    project_summary.add(analysis)
                    results.append(analysis)
                gitAnalysis = analyzer.analyze_file(file_path, directory)
                if gitAnalysis:
                    blamResults.append(gitAnalysis)
    project_summary.update_file_percentages()  # 更新各语言的文件百分比
    return project_summary, results, blamResults

def should_exclude(dir_name, analyzer):
    # 定义需要排除的目录名称
    exclude_dirs = {'.git', '__pycache__'}
    # 排除以 '.' 开头的隐藏文件夹，以及在排除列表中的文件夹
    return dir_name.startswith('.') or dir_name in exclude_dirs


def do_analyze(repo_path):
    
    summary, file_stats = analyze_directory(repo_path)
    return


class CodeCommitAnalyzer:
    def __init__(self, repo_path: str):
        self.repo = Repo(repo_path)
        self._commits_cache = {}

    def _get_commit_details(self, hexsha: str) -> Dict[str, Any]:
        """缓存提交信息以提升性能"""
        if hexsha not in self._commits_cache:
            commit = self.repo.commit(hexsha)
            self._commits_cache[hexsha] = {
                "hash": hexsha,
                "author": commit.author.name,
                "email": commit.author.email,
                "date": datetime.fromtimestamp(commit.authored_date).isoformat(),
                "message": commit.message.strip(),
                "jira" : extract_jira_id(commit.message.strip())
            }
        return self._commits_cache[hexsha]

    def analyze_file(self, file_path: str, root_path: str) -> List[Dict[str, Any]]:
        """分析文件并返回结构化数据"""
        results = []
        try:
            blame_data: List[Tuple[Commit, List[str]]] = self.repo.blame(
                rev="HEAD",
                file=file_path,
                incremental=False,
                porcelain=True
            )
        except Exception as e: return results

        for commit, lines in blame_data:
            commit_detail = self._get_commit_details(commit.hexsha)

            results.append({
                "file": os.path.relpath(file_path, root_path) ,  # 转换为1-based行号
                "code": "".join(lines) if isinstance(lines, list) and all(isinstance(line, str) for line in lines) else "",
                **commit_detail
            })

        return results

    @staticmethod
    def print_results(results: List[Dict[str, Any]]):
        """格式化打印结果"""
        for line in results:
            print(f"{line['file']:4} | {line['code']:40} | "
                  f"{line['author'][:15]:15} | {line['date'][:10]} | "
                  f"{line['message'][:50]}")


def csvs_to_json(input_files):
    """
    多文件合并核心版（不处理日期）
    
    参数：
        input_files: 支持列表/目录/通配符
        output_json: 输出路径（可选）
    """

    # 内存优化读取（保留原始数据）
    chunks = []
    for file in input_files:
        reader = pd.read_csv(file, 
                           dtype=str,         # 保持原始格式
                           keep_default_na=False,  # 禁用自动NA检测
                           chunksize=10000)   # 分块读取优化内存
        chunks.extend(reader)
    
    # 合并数据集
    combined_df = pd.concat(chunks, ignore_index=True)


    # 生成标准化JSON
    json_data = combined_df.to_json(orient='records', 
                                   indent=2,
                                   force_ascii=False)
    

    jira = json.loads(json_data)

    filtered_data = [
    {k: v for k, v in item.items() if v not in (None, "")}
    for item in jira
     ]

    return json.dumps(filtered_data, indent=2)


def extract_jira_id(commit_message):
    # 匹配方括号中的JIRA ID（格式如：XXX-1234）
    match = re.search(r'\[([A-Z]+-\d+)\]', commit_message)
    return match.group(1) if match else None


if __name__ == "__main__":
    
    output = "C:\\Users\\scyu\\Desktop\\ffl\\"
    
    # 修改为你存放 C# 代码的目录
    directory_path = "C:\\Users\\scyu\\Desktop\\code\\ffl"
    jirastr = csvs_to_json(["C:\\Users\\scyu\\Desktop\\code\\ffl_0.csv", "C:\\Users\\scyu\\Desktop\\code\\ffl_1.csv", "C:\\Users\\scyu\\Desktop\\code\\ffl_2.csv", "C:\\Users\\scyu\\Desktop\\code\\ffl_3.csv"])
    jira = json.loads(jirastr)
    with open(output + "jiradata.json", 'w', encoding='utf-8') as f:
            f.write(jirastr)



    # 统计各文件详细数据
    summary, file_stats, blamResults  = analyze_directory(directory_path)
    

    try:
        blamstr = json.dumps(blamResults, indent=2)
        with open(output + "blamdata.json", 'w', encoding='utf-8') as f:
            f.write(blamstr)

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)



    
    print("各文件详细统计：")
    file_list = []

    for analysis in file_stats:
        extra = analysis.extra if hasattr(analysis, "extra") else {}
        file_list.append({
            "file": os.path.relpath(analysis.path, directory_path) ,
            "total_lines": analysis.source_count,
            "code_lines": analysis.code_count,
            "documentation_lines": analysis.documentation_count,
            "empty_lines": analysis.empty_count,
            "string_lines": analysis.string_count,
        })

    file_stats_str = json.dumps(file_list)
    with open(output + "filedata.json", 'w', encoding='utf-8') as f:
            f.write(file_stats_str)
    
    # 使用 ProjectSummary 对整体进行统计


    lan_list = []
    print("\n总体统计摘要（基于 pygount 的 ProjectSummary）：")
    for language, lang_summary in summary.language_to_language_summary_map.items():
        lan_list.append(lang_summary)
    
    summarystr = json.dumps(lan_list, default=lambda o: o.__dict__ if isinstance(o, LanguageSummary) else str(o), indent=2)
    with open(output + "summary.json", 'w', encoding='utf-8') as f:
            f.write(summarystr)


    print("\n额外统计信息：")
    print(f"  文件总数：{summary.total_documentation_count}")
    print(f"  总行数（直接读取）：{summary.total_line_count}")
