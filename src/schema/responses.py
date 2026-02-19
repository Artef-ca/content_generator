"""
Response schemas — Pydantic models for all API responses.
"""

from typing import Optional
from pydantic import BaseModel, Field


class LogoInfo(BaseModel):
    applied: bool = False
    source: Optional[str] = None
    position: Optional[str] = None
    size: Optional[str] = None
    error: Optional[str] = None

    class Config:
        extra = "allow"


class AudioInfo(BaseModel):
    mode: str = "none"
    merged: Optional[bool] = None
    tts_language: Optional[str] = None
    tts_chars: Optional[int] = None
    uploaded_file: Optional[str] = None
    detail: Optional[str] = None
    error: Optional[str] = None

    class Config:
        extra = "allow"


class EndFrameInfo(BaseModel):
    applied: bool = False
    filename: Optional[str] = None
    auto_upgraded_model: Optional[str] = None
    error: Optional[str] = None

    class Config:
        extra = "allow"


class ImageGenerationResponse(BaseModel):
    status: str = "success"
    gcs_uri: str
    public_url: str
    model: str
    prompt_used: str
    aspect_ratio: str
    image_size: str
    logo: LogoInfo
    model_commentary: str = ""
    api_version: str = ""


class VideoGenerationResponse(BaseModel):
    status: str = "success"
    gcs_uri: str
    public_url: str
    model: str
    generation_mode: str
    prompt_used: str
    aspect_ratio: str
    resolution: str
    duration_seconds: int
    number_of_videos: int
    audio: AudioInfo
    end_frame: EndFrameInfo
    reference_images: list[str] = []
    operation: str = ""


class RefineImageResponse(BaseModel):
    status: str = "success"
    gcs_uri: str
    public_url: str
    model: str
    prompt_used: str
    original_image_source: str = ""
    model_commentary: str = ""


class RefineVideoResponse(BaseModel):
    status: str = "success"
    gcs_uri: str
    public_url: str
    model: str
    prompt_used: str
    original_video_source: str = ""
    operation: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    image_model: str
    video_models: dict
