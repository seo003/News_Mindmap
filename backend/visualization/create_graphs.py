#!/usr/bin/env python3
"""
논문용 그래프 생성 스크립트

평가 결과를 시각화하여 논문에 사용할 그래프를 생성합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# 한글 폰트 설정
import platform

def setup_korean_font():
    """한글 폰트 설정 - Windows에서 한글 폰트 경로 직접 지정"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows 한글 폰트 경로
        font_paths = [
            'C:/Windows/Fonts/malgun.ttf',           # 맑은 고딕
            'C:/Windows/Fonts/gulim.ttc',            # 굴림
            'C:/Windows/Fonts/batang.ttc',           # 바탕
            'C:/Windows/Fonts/NanumGothic.ttf',      # 나눔고딕 (설치된 경우)
        ]
        
        # 폰트 파일 찾기
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
        
        # 폰트 경로를 찾지 못한 경우, 시스템 폰트 목록에서 찾기
        font_list = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Batang']
        
        for font_name in korean_fonts:
            if font_name in font_list:
                plt.rcParams['font.family'] = font_name
                print(f"✅ 한글 폰트 설정: {font_name}")
                break
        else:
            # 최후의 수단: 맑은 고딕 강제 설정
            plt.rcParams['font.family'] = 'Malgun Gothic'
            print("⚠️ 폰트 파일을 찾지 못했습니다. 'Malgun Gothic'으로 설정합니다.")
    
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
        print("✅ 한글 폰트 설정: AppleGothic")
    
    else:  # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
        print("✅ 한글 폰트 설정: NanumGothic")
    
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    
    # 폰트 설정 확인
    test_font = plt.rcParams['font.family']
    print(f"📝 현재 폰트: {test_font}")

# 폰트 설정 실행
setup_korean_font()

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


class GraphGenerator:
    """논문용 그래프 생성 클래스"""
    
    def __init__(self, output_dir: str = "graphs"):
        """
        초기화
        
        Args:
            output_dir: 그래프 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 한글 폰트 속성 설정
        self.korean_font = self._get_korean_font()
        
        # 방법별 색상 정의
        self.method_colors = {
            'simple': '#FF6B6B',      # 빨강
            'tfidf': '#4ECDC4',       # 청록
            'fasttext': '#45B7D1',    # 파랑
            'news_analyzer': '#96CEB4' # 녹색
        }
        
        # 방법별 한글 이름
        self.method_names = {
            'simple': '빈도수',
            'tfidf': 'TF-IDF',
            'fasttext': 'FastText',
            'news_analyzer': 'HDBSCAN'
        }
    
    def _get_korean_font(self):
        """한글 폰트 FontProperties 반환"""
        system = platform.system()
        
        if system == 'Windows':
            # Windows 한글 폰트 경로
            font_paths = [
                'C:/Windows/Fonts/malgun.ttf',           # 맑은 고딕
                'C:/Windows/Fonts/gulim.ttc',            # 굴림
                'C:/Windows/Fonts/batang.ttc',           # 바탕
            ]
            
            for path in font_paths:
                if os.path.exists(path):
                    try:
                        return fm.FontProperties(fname=path)
                    except Exception as e:
                        print(f"⚠️ 폰트 파일 로드 실패 ({path}): {e}")
                        continue
            
            # 폰트 파일을 찾지 못한 경우 시스템 폰트 사용
            # 시스템 폰트 목록에서 찾기
            font_list = [f.name for f in fm.fontManager.ttflist]
            korean_fonts = ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Batang']
            
            for font_name in korean_fonts:
                if font_name in font_list:
                    return fm.FontProperties(family=font_name)
            
            # 최후의 수단
            return fm.FontProperties(family='Malgun Gothic')
        
        elif system == 'Darwin':  # macOS
            return fm.FontProperties(family='AppleGothic')
        
        else:  # Linux
            return fm.FontProperties(family='NanumGothic')
    
    def load_evaluation_results(self, results_file: Optional[str] = None) -> Dict:
        """
        평가 결과 로드
        
        Args:
            results_file: JSON 결과 파일 경로 (None이면 자동으로 찾거나 샘플 데이터 사용)
            
        Returns:
            dict: 평가 결과 딕셔너리
        """
        # results_file이 지정되지 않았으면 자동으로 찾기
        if results_file is None:
            # backend/data/evaluation_results/evaluation_results_combined.json 찾기
            backend_dir = Path(__file__).parent.parent
            default_file = backend_dir / "data" / "evaluation_results" / "evaluation_results_combined.json"
            
            if default_file.exists():
                results_file = str(default_file)
                print(f"📁 자동으로 발견된 결과 파일 사용: {results_file}")
            else:
                print(f"⚠️ 결과 파일을 찾을 수 없습니다: {default_file}")
                print("📊 샘플 데이터를 사용합니다.")
        
        if results_file and os.path.exists(results_file):
            print(f"📂 결과 파일 로드 중: {results_file}")
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 통합 파일인 경우 (method별로 키가 있는 경우)
            if any(key in data for key in ['simple', 'tfidf', 'fasttext', 'news_analyzer']):
                # last_updated 같은 메타데이터 제거
                graph_data = {k: v for k, v in data.items() 
                            if k in ['simple', 'tfidf', 'fasttext', 'news_analyzer']}
                print(f"✅ 통합 결과 파일 로드 완료: {len(graph_data)}개 방법")
                return graph_data
            else:
                # 개별 파일인 경우 (단일 방법 결과)
                print(f"✅ 개별 결과 파일 로드 완료")
                return data
        
        # 샘플 데이터 (실제 결과로 대체 필요)
        return {
            'simple': {
                'overall_score': {'score': 65.2},
                'clustering_quality': {
                    'silhouette_score': 0.15,
                    'calinski_harabasz_index': 45.2,
                    'davies_bouldin_index': 2.8,
                    'n_clusters': 25,
                    'noise_ratio': 0.0
                },
                'keyword_extraction': {'cluster_keyword_accuracy': 0.65},
                'topic_consistency': {'topic_consistency_score': 0.55},
                'performance': {
                    'total_processing_time': 8.5,
                    'throughput': 117.6
                }
            },
            'tfidf': {
                'overall_score': {'score': 72.5},
                'clustering_quality': {
                    'silhouette_score': 0.28,
                    'calinski_harabasz_index': 78.5,
                    'davies_bouldin_index': 2.1,
                    'n_clusters': 48,
                    'noise_ratio': 0.23
                },
                'keyword_extraction': {'cluster_keyword_accuracy': 0.72},
                'topic_consistency': {'topic_consistency_score': 0.68},
                'performance': {
                    'total_processing_time': 12.3,
                    'throughput': 81.3
                }
            },
            'fasttext': {
                'overall_score': {'score': 68.9},
                'clustering_quality': {
                    'silhouette_score': 0.22,
                    'calinski_harabasz_index': 62.3,
                    'davies_bouldin_index': 2.5,
                    'n_clusters': 35,
                    'noise_ratio': 0.15
                },
                'keyword_extraction': {'cluster_keyword_accuracy': 0.68},
                'topic_consistency': {'topic_consistency_score': 0.62},
                'performance': {
                    'total_processing_time': 15.2,
                    'throughput': 65.8
                }
            },
            'news_analyzer': {
                'overall_score': {'score': 85.3},
                'clustering_quality': {
                    'silhouette_score': 0.42,
                    'calinski_harabasz_index': 125.8,
                    'davies_bouldin_index': 1.5,
                    'n_clusters': 21,
                    'noise_ratio': 0.08
                },
                'keyword_extraction': {'cluster_keyword_accuracy': 0.85},
                'topic_consistency': {'topic_consistency_score': 0.82},
                'performance': {
                    'total_processing_time': 45.6,
                    'throughput': 21.9
                }
            }
        }
    
    def plot_overall_score_comparison(self, results: Dict, figsize=(10, 6)):
        """
        1. 종합 점수 비교 막대 그래프
        
        Args:
            results: 평가 결과 딕셔너리
            figsize: 그래프 크기
        """
        methods = []
        scores = []
        colors = []
        
        for method_id, method_name in self.method_names.items():
            if method_id in results:
                method_data = results[method_id]
                # overall_score가 있는지 확인
                if isinstance(method_data, dict) and 'overall_score' in method_data:
                    methods.append(method_name)
                    scores.append(method_data['overall_score']['score'])
                    colors.append(self.method_colors[method_id])
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 값 표시
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   fontproperties=self.korean_font)
        
        ax.set_ylabel('종합 점수', fontsize=14, fontweight='bold', fontproperties=self.korean_font)
        ax.set_xlabel('클러스터링 방법', fontsize=14, fontweight='bold', fontproperties=self.korean_font)
        ax.set_title('클러스터링 방법별 종합 점수 비교', fontsize=16, fontweight='bold', pad=20, fontproperties=self.korean_font)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        # x축 레이블에 한글 폰트 적용
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.korean_font)
        
        try:
            plt.tight_layout()
        except Exception:
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95)
        plt.savefig(self.output_dir / '1_overall_score_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '1_overall_score_comparison.pdf', bbox_inches='tight')
        print(f"✅ 저장 완료: {self.output_dir / '1_overall_score_comparison.png'}")
        plt.close()
    
    def plot_radar_chart(self, results: Dict, figsize=(10, 10)):
        """
        2. 평가 지표별 레이더 차트 (4개 방법)
        
        Args:
            results: 평가 결과 딕셔너리
            figsize: 그래프 크기
        """
        # 평가 지표 정의
        categories = [
            '실루엣\n점수',
            'CH Index',
            'DB Index',
            '키워드\n정확도',
            'Topic\nConsistency',
            '처리 시간',
            '처리량'
        ]
        
        # 정규화 함수 (각 지표를 0-1 범위로)
        def normalize_silhouette(score):
            return (score + 1) / 2  # -1~1 -> 0~1
        
        def normalize_ch_index(score):
            return min(1.0, np.log10(score + 1) / 4) if score > 0 else 0
        
        def normalize_db_index(score):
            return max(0, 1 - (score / 5))  # 낮을수록 좋음
        
        def normalize_accuracy(score):
            return score  # 이미 0-1 범위
        
        def normalize_time(score):
            return max(0, 1 - (score - 10) / 50)  # 10초 이내면 높은 점수
        
        def normalize_throughput(score):
            return min(1.0, score / 150)  # 150 뉴스/초 이상이면 만점
        
        # 각 방법별 데이터 준비
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 원형으로 만들기
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        for method_id, method_name in self.method_names.items():
            if method_id not in results:
                continue
            
            method_data = results[method_id]
            # results[method_id]가 직접 데이터인 경우와 딕셔너리인 경우 모두 처리
            if isinstance(method_data, dict):
                data = method_data
            else:
                continue
            
            cq = data.get('clustering_quality', {})
            ke = data.get('keyword_extraction', {})
            tc = data.get('topic_consistency', {})
            perf = data.get('performance', {})
            
            values = [
                normalize_silhouette(cq.get('silhouette_score', 0)),
                normalize_ch_index(cq.get('calinski_harabasz_index', 0)),
                normalize_db_index(cq.get('davies_bouldin_index', 5)),
                normalize_accuracy(ke.get('cluster_keyword_accuracy', 0)),
                normalize_accuracy(tc.get('topic_consistency_score', 0)),
                normalize_time(perf.get('total_processing_time', 100)),
                normalize_throughput(perf.get('throughput', 0))
            ]
            values += values[:1]  # 원형으로 만들기
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method_name, 
                   color=self.method_colors[method_id], alpha=0.7)
            ax.fill(angles, values, alpha=0.15, color=self.method_colors[method_id])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontproperties=self.korean_font)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, prop=self.korean_font)
        ax.set_title('평가 지표별 레이더 차트', fontsize=16, fontweight='bold', pad=20, fontproperties=self.korean_font)
        
        try:
            plt.tight_layout()
        except Exception:
            plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)
        plt.savefig(self.output_dir / '2_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '2_radar_chart.pdf', bbox_inches='tight')
        print(f"✅ 저장 완료: {self.output_dir / '2_radar_chart.png'}")
        plt.close()
    
    def plot_time_vs_accuracy(self, results: Dict, figsize=(10, 7)):
        """
        3. 처리 시간 vs 정확도 산점도
        
        Args:
            results: 평가 결과 딕셔너리
            figsize: 그래프 크기
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for method_id, method_name in self.method_names.items():
            if method_id not in results:
                continue
            
            method_data = results[method_id]
            if not isinstance(method_data, dict):
                continue
            
            data = method_data
            time = data.get('performance', {}).get('total_processing_time', 0)
            accuracy = data.get('overall_score', {}).get('score', 0)
            
            ax.scatter(time, accuracy, s=300, alpha=0.7, 
                      color=self.method_colors[method_id], 
                      edgecolor='black', linewidth=2,
                      label=method_name, zorder=3)
            
            # 방법 이름 표시
            ax.annotate(method_name, (time, accuracy), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       fontproperties=self.korean_font)
        
        ax.set_xlabel('처리 시간 (초)', fontsize=14, fontweight='bold', fontproperties=self.korean_font)
        ax.set_ylabel('종합 점수', fontsize=14, fontweight='bold', fontproperties=self.korean_font)
        ax.set_title('처리 시간 vs 정확도', fontsize=16, fontweight='bold', pad=20, fontproperties=self.korean_font)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best', prop=self.korean_font)
        
        try:
            plt.tight_layout()
        except Exception:
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95)
        plt.savefig(self.output_dir / '3_time_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '3_time_vs_accuracy.pdf', bbox_inches='tight')
        print(f"✅ 저장 완료: {self.output_dir / '3_time_vs_accuracy.png'}")
        plt.close()
    
    def plot_cluster_count_comparison(self, results: Dict, figsize=(10, 6)):
        """
        4. 클러스터 수 비교 막대 그래프
        
        Args:
            results: 평가 결과 딕셔너리
            figsize: 그래프 크기
        """
        methods = []
        counts = []
        colors = []
        
        for method_id, method_name in self.method_names.items():
            if method_id in results:
                method_data = results[method_id]
                if isinstance(method_data, dict) and 'clustering_quality' in method_data:
                    methods.append(method_name)
                    counts.append(method_data['clustering_quality'].get('n_clusters', 0))
                    colors.append(self.method_colors[method_id])
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(methods, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 값 표시
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   fontproperties=self.korean_font)
        
        ax.set_ylabel('클러스터 수', fontsize=14, fontweight='bold', fontproperties=self.korean_font)
        ax.set_xlabel('클러스터링 방법', fontsize=14, fontweight='bold', fontproperties=self.korean_font)
        ax.set_title('클러스터링 방법별 클러스터 수 비교', fontsize=16, fontweight='bold', pad=20, fontproperties=self.korean_font)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        # x축 레이블에 한글 폰트 적용
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.korean_font)
        
        try:
            plt.tight_layout()
        except Exception:
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95)
        plt.savefig(self.output_dir / '4_cluster_count_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '4_cluster_count_comparison.pdf', bbox_inches='tight')
        print(f"✅ 저장 완료: {self.output_dir / '4_cluster_count_comparison.png'}")
        plt.close()
    
    def plot_noise_ratio_comparison(self, results: Dict, figsize=(10, 6)):
        """
        5. 노이즈 비율 비교 막대 그래프
        
        Args:
            results: 평가 결과 딕셔너리
            figsize: 그래프 크기
        """
        methods = []
        ratios = []
        colors = []
        
        for method_id, method_name in self.method_names.items():
            if method_id in results:
                method_data = results[method_id]
                if isinstance(method_data, dict) and 'clustering_quality' in method_data:
                    methods.append(method_name)
                    ratios.append(method_data['clustering_quality'].get('noise_ratio', 0) * 100)
                    colors.append(self.method_colors[method_id])
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(methods, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 값 표시
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{ratio:.1f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   fontproperties=self.korean_font)
        
        ax.set_ylabel('노이즈 비율 (%)', fontsize=14, fontweight='bold', fontproperties=self.korean_font)
        ax.set_xlabel('클러스터링 방법', fontsize=14, fontweight='bold', fontproperties=self.korean_font)
        ax.set_title('클러스터링 방법별 노이즈 비율 비교', fontsize=16, fontweight='bold', pad=20, fontproperties=self.korean_font)
        
        # ylim 설정 개선 (0이거나 비어있는 경우 처리)
        if ratios:
            max_ratio = max(ratios)
            if max_ratio > 0:
                ax.set_ylim(0, max_ratio * 1.2)
            else:
                # 모든 값이 0인 경우 기본 범위 설정
                ax.set_ylim(0, 5)
        else:
            ax.set_ylim(0, 30)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        # x축 레이블에 한글 폰트 적용
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.korean_font)
        
        try:
            plt.tight_layout()
        except Exception:
            # tight_layout 실패 시 수동으로 여백 조정
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95)
        plt.savefig(self.output_dir / '5_noise_ratio_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '5_noise_ratio_comparison.pdf', bbox_inches='tight')
        print(f"✅ 저장 완료: {self.output_dir / '5_noise_ratio_comparison.png'}")
        plt.close()
    
    def plot_silhouette_boxplot(self, results: Dict, figsize=(10, 6)):
        """
        6. 실루엣 점수 비교 박스 플롯
        
        Args:
            results: 평가 결과 딕셔너리
            figsize: 그래프 크기
        """
        # 여러 번 실행한 결과를 가정 (실제로는 여러 실행 결과가 필요)
        # 여기서는 단일 값으로 박스 플롯 대신 막대 그래프로 대체
        methods = []
        scores = []
        colors = []
        
        for method_id, method_name in self.method_names.items():
            if method_id in results:
                method_data = results[method_id]
                if isinstance(method_data, dict) and 'clustering_quality' in method_data:
                    methods.append(method_name)
                    scores.append(method_data['clustering_quality'].get('silhouette_score', 0))
                    colors.append(self.method_colors[method_id])
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 값 표시
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold',
                   fontproperties=self.korean_font)
        
        ax.set_ylabel('실루엣 점수', fontsize=14, fontweight='bold', fontproperties=self.korean_font)
        ax.set_xlabel('클러스터링 방법', fontsize=14, fontweight='bold', fontproperties=self.korean_font)
        ax.set_title('클러스터링 방법별 실루엣 점수 비교', fontsize=16, fontweight='bold', pad=20, fontproperties=self.korean_font)
        ax.set_ylim(-0.2, 0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        # x축 레이블에 한글 폰트 적용
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.korean_font)
        
        try:
            plt.tight_layout()
        except Exception:
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95)
        plt.savefig(self.output_dir / '6_silhouette_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '6_silhouette_comparison.pdf', bbox_inches='tight')
        print(f"✅ 저장 완료: {self.output_dir / '6_silhouette_comparison.png'}")
        plt.close()
    
    def plot_comprehensive_comparison(self, results: Dict, figsize=(16, 10)):
        """
        7. 종합 비교 그래프 (서브플롯)
        
        Args:
            results: 평가 결과 딕셔너리
            figsize: 그래프 크기
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('클러스터링 방법 종합 비교', fontsize=18, fontweight='bold', y=0.995, fontproperties=self.korean_font)
        
        methods = []
        colors_list = []
        scores = []
        
        for m in self.method_names.keys():
            if m in results:
                method_data = results[m]
                if isinstance(method_data, dict) and 'overall_score' in method_data:
                    methods.append(self.method_names[m])
                    colors_list.append(self.method_colors[m])
                    scores.append(method_data['overall_score']['score'])
        axes[0, 0].bar(methods, scores, color=colors_list, alpha=0.8, edgecolor='black')
        axes[0, 0].set_title('종합 점수', fontsize=12, fontweight='bold', fontproperties=self.korean_font)
        axes[0, 0].set_ylabel('점수', fontproperties=self.korean_font)
        axes[0, 0].grid(axis='y', alpha=0.3)
        for label in axes[0, 0].get_xticklabels():
            label.set_fontproperties(self.korean_font)
        
        # 2. 클러스터 수
        cluster_counts = []
        for m in self.method_names.keys():
            if m in results:
                method_data = results[m]
                if isinstance(method_data, dict) and 'clustering_quality' in method_data:
                    cluster_counts.append(method_data['clustering_quality'].get('n_clusters', 0))
        axes[0, 1].bar(methods, cluster_counts, color=colors_list, alpha=0.8, edgecolor='black')
        axes[0, 1].set_title('클러스터 수', fontsize=12, fontweight='bold', fontproperties=self.korean_font)
        axes[0, 1].set_ylabel('개수', fontproperties=self.korean_font)
        axes[0, 1].grid(axis='y', alpha=0.3)
        for label in axes[0, 1].get_xticklabels():
            label.set_fontproperties(self.korean_font)
        
        # 3. 노이즈 비율
        noise_ratios = []
        for m in self.method_names.keys():
            if m in results:
                method_data = results[m]
                if isinstance(method_data, dict) and 'clustering_quality' in method_data:
                    noise_ratios.append(method_data['clustering_quality'].get('noise_ratio', 0) * 100)
        axes[0, 2].bar(methods, noise_ratios, color=colors_list, alpha=0.8, edgecolor='black')
        axes[0, 2].set_title('노이즈 비율', fontsize=12, fontweight='bold', fontproperties=self.korean_font)
        axes[0, 2].set_ylabel('비율 (%)', fontproperties=self.korean_font)
        axes[0, 2].grid(axis='y', alpha=0.3)
        for label in axes[0, 2].get_xticklabels():
            label.set_fontproperties(self.korean_font)
        
        # 4. 실루엣 점수
        silhouette_scores = []
        for m in self.method_names.keys():
            if m in results:
                method_data = results[m]
                if isinstance(method_data, dict) and 'clustering_quality' in method_data:
                    silhouette_scores.append(method_data['clustering_quality'].get('silhouette_score', 0))
        axes[1, 0].bar(methods, silhouette_scores, color=colors_list, alpha=0.8, edgecolor='black')
        axes[1, 0].set_title('실루엣 점수', fontsize=12, fontweight='bold', fontproperties=self.korean_font)
        axes[1, 0].set_ylabel('점수', fontproperties=self.korean_font)
        axes[1, 0].grid(axis='y', alpha=0.3)
        for label in axes[1, 0].get_xticklabels():
            label.set_fontproperties(self.korean_font)
        
        # 5. 처리 시간
        processing_times = []
        for m in self.method_names.keys():
            if m in results:
                method_data = results[m]
                if isinstance(method_data, dict) and 'performance' in method_data:
                    processing_times.append(method_data['performance'].get('total_processing_time', 0))
        axes[1, 1].bar(methods, processing_times, color=colors_list, alpha=0.8, edgecolor='black')
        axes[1, 1].set_title('처리 시간', fontsize=12, fontweight='bold', fontproperties=self.korean_font)
        axes[1, 1].set_ylabel('시간 (초)', fontproperties=self.korean_font)
        axes[1, 1].grid(axis='y', alpha=0.3)
        for label in axes[1, 1].get_xticklabels():
            label.set_fontproperties(self.korean_font)
        
        # 6. 처리량
        throughputs = []
        for m in self.method_names.keys():
            if m in results:
                method_data = results[m]
                if isinstance(method_data, dict) and 'performance' in method_data:
                    throughputs.append(method_data['performance'].get('throughput', 0))
        axes[1, 2].bar(methods, throughputs, color=colors_list, alpha=0.8, edgecolor='black')
        axes[1, 2].set_title('처리량', fontsize=12, fontweight='bold', fontproperties=self.korean_font)
        axes[1, 2].set_ylabel('뉴스/초', fontproperties=self.korean_font)
        axes[1, 2].grid(axis='y', alpha=0.3)
        for label in axes[1, 2].get_xticklabels():
            label.set_fontproperties(self.korean_font)
        
        try:
            plt.tight_layout()
        except Exception:
            plt.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.98, hspace=0.3, wspace=0.3)
        plt.savefig(self.output_dir / '7_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '7_comprehensive_comparison.pdf', bbox_inches='tight')
        print(f"✅ 저장 완료: {self.output_dir / '7_comprehensive_comparison.png'}")
        plt.close()
    
    def generate_all_graphs(self, results_file: Optional[str] = None):
        """
        모든 그래프 생성
        
        Args:
            results_file: 평가 결과 JSON 파일 경로
        """
        print("📊 논문용 그래프 생성 시작...")
        print("=" * 60)
        
        results = self.load_evaluation_results(results_file)
        
        print("\n1️⃣ 종합 점수 비교 막대 그래프 생성 중...")
        self.plot_overall_score_comparison(results)
        
        print("\n2️⃣ 평가 지표별 레이더 차트 생성 중...")
        self.plot_radar_chart(results)
        
        print("\n3️⃣ 처리 시간 vs 정확도 산점도 생성 중...")
        self.plot_time_vs_accuracy(results)
        
        print("\n4️⃣ 클러스터 수 비교 막대 그래프 생성 중...")
        self.plot_cluster_count_comparison(results)
        
        print("\n5️⃣ 노이즈 비율 비교 막대 그래프 생성 중...")
        self.plot_noise_ratio_comparison(results)
        
        print("\n6️⃣ 실루엣 점수 비교 그래프 생성 중...")
        self.plot_silhouette_boxplot(results)
        
        print("\n7️⃣ 종합 비교 그래프 생성 중...")
        self.plot_comprehensive_comparison(results)
        
        print("\n" + "=" * 60)
        print(f"✅ 모든 그래프 생성 완료! 저장 위치: {self.output_dir.absolute()}")
        print("\n생성된 파일:")
        for i in range(1, 8):
            print(f"  - {i}_*.png (PNG 형식)")
            print(f"  - {i}_*.pdf (PDF 형식)")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='논문용 그래프 생성')
    parser.add_argument('--results', type=str, default=None,
                       help='평가 결과 JSON 파일 경로 (없으면 샘플 데이터 사용)')
    parser.add_argument('--output', type=str, default='graphs',
                       help='그래프 저장 디렉토리 (기본값: graphs)')
    
    args = parser.parse_args()
    
    generator = GraphGenerator(output_dir=args.output)
    generator.generate_all_graphs(results_file=args.results)


if __name__ == "__main__":
    main()

