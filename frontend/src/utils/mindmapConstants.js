/**
 * 마인드맵 관련 상수 정의
 */

// 화면 크기 브레이크포인트
export const BREAKPOINTS = {
  VERY_SMALL: 320,
  SMALL: 480,
  MEDIUM: 768,
  LARGE: 1024,
};

// 줌 레벨 설정
export const ZOOM_CONFIG = {
  MIN: 0.1,
  MAX: 10,
  MULTIPLIER: 1.2,
  INITIAL: {
    [BREAKPOINTS.VERY_SMALL]: 1.9,
    [BREAKPOINTS.SMALL]: 2.2,
    [BREAKPOINTS.MEDIUM]: 2.6,
    [BREAKPOINTS.LARGE]: 2.9,
    DEFAULT: 3.4,
  },
};

// Force 그래프 설정
export const FORCE_CONFIG = {
  CHARGE_STRENGTH: {
    VERY_SMALL: -250,
    SMALL: -300,
    MEDIUM: -350,
    DEFAULT: -500,
  },
  LINK_DISTANCE: {
    VERY_SMALL: 50,
    SMALL: 60,
    MEDIUM: 70,
    DEFAULT: 100,
  },
  CENTER_STRENGTH: {
    MOBILE: 0.2,
    DESKTOP: 0.1,
  },
  INITIAL_SPREAD_RADIUS: {
    SMALL: 700,
    MEDIUM: 600,
    DEFAULT: 500,
  },
};

// 노드 레벨 상수
export const NODE_LEVELS = {
  CENTRAL: 0,
  MAJOR: 1,
  MIDDLE: 2,
};

// 노드 크기 설정
export const NODE_SIZE_CONFIG = {
  // 중앙 노드 (Level 0)
  CENTRAL: {
    FONT_SIZE: {
      VERY_SMALL: 9,
      SMALL: 11,
      MEDIUM: 13,
      DEFAULT: 15,
    },
    HEIGHT: {
      VERY_SMALL: 16,
      SMALL: 20,
      MEDIUM: 24,
      DEFAULT: 28,
    },
    WIDTH_MULTIPLIER: {
      VERY_SMALL: 0.6,
      SMALL: 0.7,
      MEDIUM: 0.8,
      DEFAULT: 1.0,
    },
    MIN_WIDTH: 40,
  },
  // 대분류 노드 (Level 1)
  MAJOR: {
    BASE_FONT_SIZE: {
      VERY_SMALL: 4,
      SMALL: 5,
      MEDIUM: 6,
      DEFAULT: 7,
    },
    BASE_HEIGHT: {
      VERY_SMALL: 10,
      SMALL: 12,
      MEDIUM: 14,
      DEFAULT: 16,
    },
    WIDTH_MULTIPLIER: {
      VERY_SMALL: 0.6,
      SMALL: 0.7,
      MEDIUM: 0.8,
      DEFAULT: 1.0,
    },
    MIN_WIDTH: 25,
    SIZE_MULTIPLIER: {
      MIN: 0.8,
      MAX: 2.0,
      BASE: 0.8,
      RANGE: 1.2,
    },
    HEIGHT_MULTIPLIER: 1.2,
    FONT_SIZE_MIN: 0.7,
  },
  // 중분류 노드 (Level 2)
  MIDDLE: {
    BASE_FONT_SIZE: {
      VERY_SMALL: 4,
      SMALL: 5,
      MEDIUM: 6,
      DEFAULT: 8,
    },
    BASE_HEIGHT: {
      VERY_SMALL: 10,
      SMALL: 12,
      MEDIUM: 14,
      DEFAULT: 18,
    },
    WIDTH_MULTIPLIER: {
      VERY_SMALL: 0.6,
      SMALL: 0.7,
      MEDIUM: 0.8,
      DEFAULT: 1.0,
    },
    MIN_WIDTH: 36,
    SIZE_MULTIPLIER: {
      MIN: 0.7,
      MAX: 1.4,
      BASE: 0.7,
      RANGE: 0.7,
    },
    FONT_SIZE_MIN: 0.8,
  },
};

// 노드 색상 설정
export const NODE_COLORS = {
  CENTRAL: {
    FILL: '#1e3a8a',
    BORDER: '#1e40af',
    TEXT: 'white',
  },
  MAJOR: {
    DARK: { R: 59, G: 130, B: 246 },
    LIGHT: { R: 147, G: 197, B: 253 },
    TEXT_DARK: '#1e40af',
    TEXT_LIGHT: 'white',
    RANK_THRESHOLD: 0.7,
    BORDER_MULTIPLIER: 0.8,
  },
  MIDDLE: {
    DARK: { R: 147, G: 197, B: 253 },
    LIGHT: { R: 219, G: 234, B: 254 },
    DEFAULT_FILL: '#93c5fd',
    DEFAULT_BORDER: '#60a5fa',
    TEXT_DARK: '#1e40af',
    TEXT_LIGHT: 'white',
    RANK_THRESHOLD: 0.7,
    BORDER_MULTIPLIER: 0.8,
  },
  DEFAULT: {
    FILL: '#dbeafe',
    BORDER: '#93c5fd',
    TEXT: '#1e40af',
  },
};

// 링크 색상 설정
export const LINK_COLORS = {
  GRADIENT_START: 'rgba(59, 130, 246, 0.8)',
  GRADIENT_END: 'rgba(37, 99, 235, 0.6)',
};

// 링크 두께 설정
export const LINK_WIDTH = {
  MAJOR: 2,
  MIDDLE: 1.5,
  DEFAULT: 1,
};

// 클릭 영역 배율
export const CLICK_AREA_SCALE = 1.2;

// 타원형 비율
export const ELLIPSE_RATIO = 0.8;

// 애니메이션 시간
export const ANIMATION_DURATION = {
  CENTER_AT: 1000,
  ZOOM_RETRY: 100,
  ZOOM_DELAY: 200,
};

// 뉴스 제목 최대 길이
export const NEWS_TITLE_MAX_LENGTH = 50;

// 최소 뉴스 개수 (중분류 표시 조건)
export const MIN_NEWS_COUNT_FOR_DISPLAY = 2;

