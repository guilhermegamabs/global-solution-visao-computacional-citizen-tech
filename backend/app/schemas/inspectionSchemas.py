from pydantic import BaseModel


class TileResult(BaseModel):
    row: int
    column: int
    probability: float
    flagged: bool


class ImageInspection(BaseModel):
    label: str
    crackProbability: float
    confidence: float
    threshold: float
    needsInspection: bool
    flaggedTiles: int
    totalTiles: int
    tiles: list[TileResult]
    annotatedImage: str
    latencyMs: float
    imageWidth: int
    imageHeight: int
    disclaimer: str


class TimelinePoint(BaseModel):
    frameIndex: int
    timeSeconds: float
    crackProbability: float
    flagged: bool


class FramePreview(BaseModel):
    frameIndex: int
    timeSeconds: float
    crackProbability: float
    image: str


class VideoInspection(BaseModel):
    needsInspection: bool
    threshold: float
    framesAnalyzed: int
    framesFlagged: int
    flaggedRatio: float
    averageLatencyMs: float
    processingFps: float
    sourceFps: float
    durationSeconds: float
    timeline: list[TimelinePoint]
    previews: list[FramePreview]
    disclaimer: str


class ModelInfo(BaseModel):
    name: str
    inputSize: int
    defaultThreshold: float
    tileGrid: int
    available: bool
    metrics: dict | None = None
