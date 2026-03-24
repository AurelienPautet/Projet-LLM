import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import fitz
from langchain_core.tools import tool
from sqlmodel import Session, select
from pgvector.sqlalchemy import Vector
from sqlalchemy import cast

from db.db import engine, Experience, ExperienceBase, PersonalInfo, Offer
from graph.embedding import createEmbeddingFromText
from llmUtils import schemaToEmbeddingText



def compileLatexToPdf(latexCode: str, outputName: str = "cv") -> str:
    latexBinary = shutil.which("pdflatex")
    if latexBinary is None:
        return "Error: pdflatex is not installed or not available in PATH."

    cleanName = re.sub(r"[^a-zA-Z0-9.-]", "-", outputName).strip(".-") or "cv"
    if cleanName.lower().endswith(".pdf"):
        cleanName = cleanName[:-4]

    outputDir = os.path.abspath("generatedPdfs")
    os.makedirs(outputDir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="latexBuild-") as tempDir:
        texPath = os.path.join(tempDir, f"{cleanName}.tex")
        with open(texPath, "w", encoding="utf-8") as f:
            f.write(latexCode)

        run = subprocess.run(
            [latexBinary, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tempDir}", texPath],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=30, check=False,
        )

        pdfPath = os.path.join(tempDir, f"{cleanName}.pdf")
        if run.returncode != 0 or not os.path.exists(pdfPath):
            latexLog = run.stdout.decode("utf-8", errors="replace").strip() if run.stdout else "Unknown LaTeX compilation failure."
            if "File `moderncv.cls' not found" in latexLog:
                return "Error: moderncv is not installed in your LaTeX distribution. Install moderncv and retry."
            if "File `fontawesome5.sty' not found" in latexLog:
                return "Error: fontawesome5 package is missing. Install it and retry."
            return f"Error: LaTeX compilation failed.\n{latexLog}"

        finalPdfPath = os.path.join(outputDir, f"{cleanName}.pdf")
        shutil.copyfile(pdfPath, finalPdfPath)
        return f"PDF generated successfully: {finalPdfPath}"




def formatOfferSummary(dbOffer: Offer) -> str:
    source = dbOffer.offerSource or "N/A"
    cvLength = len(dbOffer.cvOutput or "")
    coverLength = len(dbOffer.coverLetterOutput or "")
    return (
        f"id={dbOffer.id} | source={source} | "
        f"cvChars={cvLength} | coverLetterChars={coverLength} | offerText={dbOffer.offerText}"
    )


def updateOfferCvOutput(offerId: int, cvOutput: str) -> Offer:
    cleanCvOutput = (cvOutput or "").strip()
    if not cleanCvOutput:
        raise ValueError("cvOutput cannot be empty")
    with Session(engine) as session:
        dbOffer = session.get(Offer, offerId)
        if dbOffer is None:
            raise ValueError(f"no offer found with id={offerId}")
        dbOffer.cvOutput = cleanCvOutput
        dbOffer.cvVersion += 1
        dbOffer.updatedAt = datetime.utcnow()
        session.add(dbOffer)
        session.commit()
        session.refresh(dbOffer)
        return dbOffer


def updateOfferCoverLetterOutput(offerId: int, coverLetterOutput: str) -> Offer:
    cleanOutput = (coverLetterOutput or "").strip()
    if not cleanOutput:
        raise ValueError("coverLetterOutput cannot be empty")
    with Session(engine) as session:
        dbOffer = session.get(Offer, offerId)
        if dbOffer is None:
            raise ValueError(f"no offer found with id={offerId}")
        dbOffer.coverLetterOutput = cleanOutput
        dbOffer.coverLetterVersion += 1
        dbOffer.updatedAt = datetime.utcnow()
        session.add(dbOffer)
        session.commit()
        session.refresh(dbOffer)
        return dbOffer


@tool
def addExperience(experience: ExperienceBase) -> str:
    """Add a professional experience to the CV database. Best practice: check if the experience already exists using searchExperiences before adding to avoid duplicates."""
    try:
        embedding = createEmbeddingFromText(schemaToEmbeddingText(experience))
        with Session(engine) as session:
            db_experience = Experience.model_validate(experience)
            db_experience.embedding = embedding
            session.add(db_experience)
            session.commit()
            session.refresh(db_experience)
        return f"Experience added: {db_experience.title} at {db_experience.company_or_institution or 'N/A'}"
    except Exception as exc:
        return f"Error: addExperience failed: {exc}"


@tool
def upsertPersonalInfo(fieldName: str, fieldValue: str) -> str:
    """Create or update one personal CV information field (for example email, phone, linkedin, github, summary, address)."""
    try:
        cleanName = fieldName.strip()
        cleanValue = fieldValue.strip()
        if not cleanName:
            return "Error: fieldName cannot be empty"
        if not cleanValue:
            return "Error: fieldValue cannot be empty"
        with Session(engine) as session:
            existing = session.exec(select(PersonalInfo).where(PersonalInfo.fieldName == cleanName)).first()
            if existing is None:
                session.add(PersonalInfo(fieldName=cleanName, fieldValue=cleanValue))
                session.commit()
                return f"Personal info added: {cleanName}"
            existing.fieldValue = cleanValue
            session.add(existing)
            session.commit()
            return f"Personal info updated: {cleanName}"
    except Exception as exc:
        return f"Error: upsertPersonalInfo failed: {exc}"


@tool
def getPersonalInfo(fieldName: str) -> str:
    """Retrieve one personal CV information field by name."""
    try:
        cleanName = fieldName.strip()
        if not cleanName:
            return "Error: fieldName cannot be empty"
        with Session(engine) as session:
            info = session.exec(select(PersonalInfo).where(PersonalInfo.fieldName == cleanName)).first()
        if info is None:
            return f"No personal info found for field: {cleanName}"
        return f"{info.fieldName}: {info.fieldValue}"
    except Exception as exc:
        return f"Error: getPersonalInfo failed: {exc}"


@tool
def getAllPersonalInfo() -> str:
    """Retrieve all personal CV information fields currently stored in database."""
    try:
        with Session(engine) as session:
            infos = session.exec(select(PersonalInfo)).all()
        if not infos:
            return "No personal info found."
        return "\n".join(f"{i.fieldName}: {i.fieldValue}" for i in sorted(infos, key=lambda x: x.fieldName.lower()))
    except Exception as exc:
        return f"Error: getAllPersonalInfo failed: {exc}"


@tool
def searchExperiences(query: str, limit: int = 5) -> str:
    """Search experiences semantically using a natural language query. Returns the most relevant experiences."""
    try:
        queryEmbedding = createEmbeddingFromText(query)
        with Session(engine) as session:
            results = session.exec(
                select(Experience)
                .order_by(Experience.embedding.op("<=>") (cast(queryEmbedding, Vector(3072))))
                .limit(limit)
            ).all()
        if not results:
            return "No experiences found."
        return "\n".join(
            f"[id={exp.id}] {exp.title} at {exp.company_or_institution or 'N/A'} "
            f"({exp.start_date or '?'} - {exp.end_date or '?'}) | {', '.join(exp.technos or [])}"
            for exp in results
        )
    except Exception as exc:
        return f"Error: searchExperiences failed: {exc}"


@tool
def editExperience(id: int, experience: ExperienceBase) -> str:
    """Edit an existing experience by its id. All fields will be replaced with the provided values."""
    try:
        with Session(engine) as session:
            dbExperience = session.get(Experience, id)
            if dbExperience is None:
                return f"Error: no experience found with id={id}"
            for field, value in experience.model_dump().items():
                setattr(dbExperience, field, value)
            dbExperience.embedding = createEmbeddingFromText(schemaToEmbeddingText(experience))
            session.add(dbExperience)
            session.commit()
            session.refresh(dbExperience)
        return f"Experience updated: {dbExperience.title} at {dbExperience.company_or_institution or 'N/A'}"
    except Exception as exc:
        return f"Error: editExperience failed: {exc}"


@tool
def getAllExperiences() -> str:
    """Get all existing experiences from the CV database. WARNING: Use sparingly as this retrieves all records. Prefer searchExperiences for targeted queries."""
    try:
        with Session(engine) as session:
            results = session.exec(select(Experience)).all()
        if not results:
            return "No experiences found."
        return "\n".join(
            f"[id={exp.id}] {exp.title} at {exp.company_or_institution or 'N/A'} "
            f"({exp.start_date or '?'} - {exp.end_date or '?'}) | {', '.join(exp.technos or [])}"
            for exp in results
        )
    except Exception as exc:
        return f"Error: getAllExperiences failed: {exc}"


@tool
def getExperienceCount() -> str:
    """Get the total number of experiences in the CV database."""
    try:
        with Session(engine) as session:
            count = len(session.exec(select(Experience)).all())
        return f"Total experiences: {count}"
    except Exception as exc:
        return f"Error: getExperienceCount failed: {exc}"


@tool
def deleteExperience(id: int) -> str:
    """Delete an experience by its id."""
    try:
        with Session(engine) as session:
            dbExperience = session.get(Experience, id)
            if dbExperience is None:
                return f"Error: no experience found with id={id}"
            session.delete(dbExperience)
            session.commit()
        return f"Experience with id={id} deleted successfully"
    except Exception as exc:
        return f"Error: deleteExperience failed: {exc}"


@tool
def loadCvFromFile(filepath: str) -> str:
    """Read a CV file (PDF, TXT, or MD) and return its text content so you can extract and add experiences from it."""
    filepath = filepath.strip().strip("'\"")
    if not os.path.exists(filepath):
        return f"Error: file not found at path: {filepath}"
    ext = filepath.lower().split('.')[-1]
    if ext == 'pdf':
        try:
            doc = fitz.open(filepath)
            return "".join(page.get_text() + "\n" for page in doc)
        except Exception as exc:
            return f"Error reading PDF: {exc}"
    elif ext in ['txt', 'md']:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as exc:
            return f"Error reading file: {exc}"
    return "Error: unsupported file format. Please provide a PDF, TXT, or MD file."


@tool
def generatePdfFromLatex(latexCode: str, outputName: str = "cv") -> str:
    """Generate a PDF file from LaTeX code and return either LaTeX errors or the generated PDF path."""
    try:
        return compileLatexToPdf(latexCode=latexCode, outputName=outputName)
    except subprocess.TimeoutExpired:
        return "Error: LaTeX compilation timed out."
    except Exception as exc:
        return f"Error: generatePdfFromLatex failed: {exc}"


@tool
def fetchWebPageContent(url: str) -> str:
    """Fetch text content from a web page URL. Requires a valid URL starting with http:// or https://."""
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"}:
            return "Error: URL must start with http:// or https://"
        request = Request(parsed.geturl(), headers={"User-Agent": "Mozilla/5.0 (CareerCopilot/1.0)"})
        with urlopen(request, timeout=20) as response:
            rawBytes = response.read()
            encoding = response.headers.get_content_charset() or "utf-8"
            html = rawBytes.decode(encoding, errors="replace")
        content = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
        content = re.sub(r"(?is)<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()
        if not content:
            return "Error: no readable text content was found on this page."
        if len(content) > 6000:
            content = content[:6000] + " ... [Content truncated to fit context limits]"
        return content
    except Exception as exc:
        return f"Error: fetchWebPageContent failed: {exc}"


@tool
def saveOffer(offerText: str, offerSource: str = "") -> str:
    """Save a new offer in database."""
    try:
        cleanOfferText = (offerText or "").strip()
        if not cleanOfferText:
            return "Error: offerText cannot be empty"
        cleanOfferSource = (offerSource or "").strip() or None
        embeddingSource = f"{cleanOfferSource}\n{cleanOfferText}" if cleanOfferSource else cleanOfferText
        offerEmbedding = createEmbeddingFromText(embeddingSource)
        with Session(engine) as session:
            dbOffer = Offer(
                offerText=cleanOfferText,
                offerSource=cleanOfferSource,
                embedding=offerEmbedding,
                updatedAt=datetime.utcnow(),
            )
            session.add(dbOffer)
            session.commit()
            session.refresh(dbOffer)
            return f"Offer saved with id={dbOffer.id}"
    except Exception as exc:
        return f"Error: saveOffer failed: {exc}"


@tool
def getOfferById(offerId: int) -> Optional[Offer]:
    """Get one offer by id from database."""
    try:
        with Session(engine) as session:
            return session.get(Offer, offerId)
    except Exception:
        return None


@tool
def searchOffers(query: str, limit: int = 5) -> str:
    """Search existing offers by semantic similarity and return matching offer ids."""
    try:
        queryEmbedding = createEmbeddingFromText(query)
        with Session(engine) as session:
            results = session.exec(
                select(Offer)
                .where(Offer.embedding.is_not(None))
                .order_by(Offer.embedding.op("<=>") (cast(queryEmbedding, Vector(3072))))
                .limit(limit)
            ).all()
        if not results:
            return "No offers found."
        return "\n".join(formatOfferSummary(dbOffer) for dbOffer in results)
    except Exception as exc:
        return f"Error: searchOffers failed: {exc}"


@tool
def getOfferBySource(sourceUrl: str) -> str:
    """Check if an offer with this source URL already exists in database."""
    try:
        cleanUrl = (sourceUrl or "").strip()
        if not cleanUrl:
            return "Error: sourceUrl cannot be empty"
        with Session(engine) as session:
            dbOffer = session.exec(select(Offer).where(Offer.offerSource == cleanUrl)).first()
        if dbOffer is None:
            return f"No offer found with source code: {cleanUrl}"
        return formatOfferSummary(dbOffer)
    except Exception as exc:
        return f"Error: getOfferBySource failed: {exc}"
