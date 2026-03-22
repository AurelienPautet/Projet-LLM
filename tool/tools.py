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

from db.db import engine, Experience, ExperienceBase, ExperienceResult, PersonalInfo, Offer, OfferStatus
from graph.embedding import createEmbeddingFromText
from llmUtils import schemaToEmbeddingText


def buildLatexDocument(cvText: str) -> str:
    escapedText = (
        cvText
        .replace("\\", r"\\textbackslash{}")
        .replace("&", r"\\&")
        .replace("%", r"\\%")
        .replace("$", r"\\$")
        .replace("#", r"\\#")
        .replace("_", r"\\_")
        .replace("{", r"\\{")
        .replace("}", r"\\}")
        .replace("~", r"\\textasciitilde{}")
        .replace("^", r"\\textasciicircum{}")
    )
    escapedText = escapedText.replace("\n", "\n\\\\\n")
    return (
        "\\documentclass[11pt,a4paper]{article}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\begin{document}\n"
        f"{escapedText}\n"
        "\\end{document}\n"
    )


def compileLatexToPdf(latexCode: str, outputName: str = "cv") -> str:
    latexBinary = shutil.which("pdflatex")
    if latexBinary is None:
        return "Error: pdflatex is not installed or not available in PATH."

    cleanName = re.sub(r"[^a-zA-Z0-9.-]", "-", outputName).strip(".-")
    if not cleanName:
        cleanName = "cv"
    if cleanName.lower().endswith(".pdf"):
        cleanName = cleanName[:-4]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    finalBaseName = f"{cleanName}_{timestamp}"

    outputDir = os.path.abspath("generatedPdfs")
    os.makedirs(outputDir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="latexBuild-") as tempDir:
        texPath = os.path.join(tempDir, f"{finalBaseName}.tex")
        with open(texPath, "w", encoding="utf-8") as texFile:
            texFile.write(latexCode)

        cmd = [
            latexBinary,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={tempDir}",
            texPath,
        ]
        run = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )

        pdfPath = os.path.join(tempDir, f"{finalBaseName}.pdf")
        if run.returncode != 0 or not os.path.exists(pdfPath):
            if isinstance(run.stdout, bytes):
                latexLog = run.stdout.decode("utf-8", errors="replace").strip()
            else:
                latexLog = str(run.stdout or "").strip()
            if not latexLog:
                latexLog = "Unknown LaTeX compilation failure."
            if "File `moderncv.cls' not found" in latexLog:
                return "Error: moderncv is not installed in your LaTeX distribution. Install moderncv and retry."
            if "File `fontawesome5.sty' not found" in latexLog:
                return "Error: fontawesome5 package is missing in your LaTeX distribution. Install it and retry."
            return f"Error: LaTeX compilation failed.\n{latexLog}"

        finalPdfPath = os.path.join(outputDir, f"{finalBaseName}.pdf")
        shutil.copyfile(pdfPath, finalPdfPath)
        return f"PDF generated successfully: {finalPdfPath}"


def extractPlainTextFromLatex(latexCode: str) -> str:
    text = latexCode
    text = re.sub(r"(?is)%.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\\begin\{[^}]+\}", " ", text)
    text = re.sub(r"\\end\{[^}]+\}", " ", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def saveOfferRecord(offerText: str, offerSource: Optional[str] = None, status: OfferStatus = OfferStatus.OFFER_COLLECTED) -> Offer:
    cleanOfferText = (offerText or "").strip()
    if not cleanOfferText:
        raise ValueError("offerText cannot be empty")
    cleanOfferSource = (offerSource or "").strip() or None
    now = datetime.utcnow()
    with Session(engine) as session:
        dbOffer = Offer(
            offerText=cleanOfferText,
            offerSource=cleanOfferSource,
            status=status,
            updatedAt=now,
        )
        session.add(dbOffer)
        session.commit()
        session.refresh(dbOffer)
        return dbOffer


def getActiveOfferRecord() -> Optional[Offer]:
    with Session(engine) as session:
        return session.exec(select(Offer).order_by(Offer.updatedAt.desc(), Offer.id.desc())).first()


def getOfferByIdRecord(offerId: int) -> Optional[Offer]:
    with Session(engine) as session:
        return session.get(Offer, offerId)


def updateOfferStatusRecord(offerId: int, status: OfferStatus) -> Offer:
    with Session(engine) as session:
        dbOffer = session.get(Offer, offerId)
        if dbOffer is None:
            raise ValueError(f"no offer found with id={offerId}")
        dbOffer.status = status
        dbOffer.updatedAt = datetime.utcnow()
        session.add(dbOffer)
        session.commit()
        session.refresh(dbOffer)
        return dbOffer


def updateOfferCvOutputRecord(offerId: int, cvOutput: str) -> Offer:
    cleanCvOutput = (cvOutput or "").strip()
    if not cleanCvOutput:
        raise ValueError("cvOutput cannot be empty")
    with Session(engine) as session:
        dbOffer = session.get(Offer, offerId)
        if dbOffer is None:
            raise ValueError(f"no offer found with id={offerId}")
        dbOffer.cvOutput = cleanCvOutput
        if dbOffer.coverLetterOutput:
            dbOffer.status = OfferStatus.COMPLETED
        else:
            dbOffer.status = OfferStatus.CV_GENERATED
        dbOffer.updatedAt = datetime.utcnow()
        session.add(dbOffer)
        session.commit()
        session.refresh(dbOffer)
        return dbOffer


def updateOfferCoverLetterOutputRecord(offerId: int, coverLetterOutput: str) -> Offer:
    cleanCoverLetterOutput = (coverLetterOutput or "").strip()
    if not cleanCoverLetterOutput:
        raise ValueError("coverLetterOutput cannot be empty")
    with Session(engine) as session:
        dbOffer = session.get(Offer, offerId)
        if dbOffer is None:
            raise ValueError(f"no offer found with id={offerId}")
        dbOffer.coverLetterOutput = cleanCoverLetterOutput
        if dbOffer.cvOutput:
            dbOffer.status = OfferStatus.COMPLETED
        else:
            dbOffer.status = OfferStatus.COVER_LETTER_GENERATED
        dbOffer.updatedAt = datetime.utcnow()
        session.add(dbOffer)
        session.commit()
        session.refresh(dbOffer)
        return dbOffer


@tool
def addExperience(experience: ExperienceBase) -> str:
    """Add a professional experience to the CV database. Best practice: check if the experience already exists using searchExperiences before adding to avoid duplicates."""
    try:
        embedding = createEmbeddingFromText(
            schemaToEmbeddingText(experience)
        )

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
        cleanFieldName = fieldName.strip()
        cleanFieldValue = fieldValue.strip()
        if not cleanFieldName:
            return "Error: fieldName cannot be empty"
        if not cleanFieldValue:
            return "Error: fieldValue cannot be empty"

        with Session(engine) as session:
            existingInfo = session.exec(
                select(PersonalInfo).where(
                    PersonalInfo.fieldName == cleanFieldName)
            ).first()

            if existingInfo is None:
                dbPersonalInfo = PersonalInfo(
                    fieldName=cleanFieldName,
                    fieldValue=cleanFieldValue,
                )
                session.add(dbPersonalInfo)
                session.commit()
                session.refresh(dbPersonalInfo)
                return f"Personal info added: {dbPersonalInfo.fieldName}"

            existingInfo.fieldValue = cleanFieldValue
            session.add(existingInfo)
            session.commit()
            session.refresh(existingInfo)
            return f"Personal info updated: {existingInfo.fieldName}"
    except Exception as exc:
        return f"Error: upsertPersonalInfo failed: {exc}"


@tool
def getPersonalInfo(fieldName: str) -> str:
    """Retrieve one personal CV information field by name."""
    try:
        cleanFieldName = fieldName.strip()
        if not cleanFieldName:
            return "Error: fieldName cannot be empty"

        with Session(engine) as session:
            existingInfo = session.exec(
                select(PersonalInfo).where(
                    PersonalInfo.fieldName == cleanFieldName)
            ).first()

        if existingInfo is None:
            return f"No personal info found for field: {cleanFieldName}"

        return f"{existingInfo.fieldName}: {existingInfo.fieldValue}"
    except Exception as exc:
        return f"Error: getPersonalInfo failed: {exc}"


@tool
def getAllPersonalInfo() -> str:
    """Retrieve all personal CV information fields currently stored in database."""
    try:
        with Session(engine) as session:
            personalInfos = session.exec(select(PersonalInfo)).all()

        if not personalInfos:
            return "No personal info found."

        sortedInfos = sorted(
            personalInfos, key=lambda info: info.fieldName.lower())
        return "\n".join([f"{info.fieldName}: {info.fieldValue}" for info in sortedInfos])
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
                .order_by(Experience.embedding.op("<=>")(cast(queryEmbedding, Vector(3072))))
                .limit(limit)
            ).all()

        if not results:
            return "No experiences found."

        out: List[str] = []
        for exp in results:
            technos = ", ".join(exp.technos or [])
            out.append(
                f"[id={exp.id}] {exp.title} at {exp.company_or_institution or 'N/A'} "
                f"({str(exp.start_date) if exp.start_date else '?'} - {str(exp.end_date) if exp.end_date else '?'}) | {technos}"
            )
        return "\n".join(out)
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

            dbExperience.embedding = createEmbeddingFromText(
                schemaToEmbeddingText(experience)
            )

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

        out: List[str] = []
        for exp in results:
            technos = ", ".join(exp.technos or [])
            out.append(
                f"[id={exp.id}] {exp.title} at {exp.company_or_institution or 'N/A'} "
                f"({str(exp.start_date) if exp.start_date else '?'} - {str(exp.end_date) if exp.end_date else '?'}) | {technos}"
            )
        return "\n".join(out)
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
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
        except Exception as exc:
            return f"Error reading PDF: {exc}"
    elif ext in ['txt', 'md']:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as exc:
            return f"Error reading file: {exc}"
    else:
        return "Error: unsupported file format. Please provide a PDF, TXT, or MD file."


@tool
def generatePdfFromLatex(latexCode: str, outputName: str = "cv") -> str:
    """Generate a PDF file from LaTeX code and return either LaTeX errors or the generated PDF path."""
    try:
        firstAttempt = compileLatexToPdf(
            latexCode=latexCode, outputName=outputName)
        if not firstAttempt.startswith("Error:"):
            return firstAttempt

        plainText = extractPlainTextFromLatex(latexCode)
        if not plainText:
            return firstAttempt

        safeLatex = buildLatexDocument(plainText)
        secondAttempt = compileLatexToPdf(
            latexCode=safeLatex, outputName=outputName)
        if secondAttempt.startswith("Error:"):
            return firstAttempt
        return secondAttempt
    except subprocess.TimeoutExpired:
        return "Error: LaTeX compilation timed out."
    except Exception as exc:
        return f"Error: generatePdfFromLatex failed: {exc}"


@tool
def fetchWebPageContent(url: str) -> str:
    """Fetch text content from a web page URL."""
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"}:
            return "Error: URL must start with http:// or https://"

        request = Request(
            parsed.geturl(),
            headers={"User-Agent": "Mozilla/5.0 (CareerCopilot/1.0)"},
        )
        with urlopen(request, timeout=20) as response:
            rawBytes = response.read()
            encoding = response.headers.get_content_charset() or "utf-8"
            html = rawBytes.decode(encoding, errors="replace")

        content = re.sub(
            r"(?is)<(script|style|noscript).*?>.*?</\\1>",
            " ",
            html,
        )
        content = re.sub(r"(?is)<[^>]+>", " ", content)
        content = re.sub(r"\\s+", " ", content).strip()
        if not content:
            return "Error: no readable text content was found on this page."

        maxChars = 12000
        if len(content) > maxChars:
            content = content[:maxChars]
        return content
    except Exception as exc:
        return f"Error: fetchWebPageContent failed: {exc}"


@tool
def saveOffer(offerText: str, offerSource: str = "") -> str:
    """Save a new offer in database and mark it as collected."""
    try:
        dbOffer = saveOfferRecord(offerText=offerText, offerSource=offerSource)
        return f"Offer saved with id={dbOffer.id} and status={dbOffer.status.value}"
    except Exception as exc:
        return f"Error: saveOffer failed: {exc}"


@tool
def getActiveOffer() -> str:
    """Get the most recently active offer from database."""
    try:
        dbOffer = getActiveOfferRecord()
        if dbOffer is None:
            return "No active offer found."
        source = dbOffer.offerSource or "N/A"
        cvLength = len(dbOffer.cvOutput or "")
        coverLength = len(dbOffer.coverLetterOutput or "")
        return (
            f"id={dbOffer.id} | status={dbOffer.status.value} | source={source} | "
            f"cvChars={cvLength} | coverLetterChars={coverLength} | offerText={dbOffer.offerText}"
        )
    except Exception as exc:
        return f"Error: getActiveOffer failed: {exc}"


@tool
def getOfferById(offerId: int) -> str:
    """Get one offer by id from database."""
    try:
        dbOffer = getOfferByIdRecord(offerId)
        if dbOffer is None:
            return f"No offer found with id={offerId}."
        source = dbOffer.offerSource or "N/A"
        cvLength = len(dbOffer.cvOutput or "")
        coverLength = len(dbOffer.coverLetterOutput or "")
        return (
            f"id={dbOffer.id} | status={dbOffer.status.value} | source={source} | "
            f"cvChars={cvLength} | coverLetterChars={coverLength} | offerText={dbOffer.offerText}"
        )
    except Exception as exc:
        return f"Error: getOfferById failed: {exc}"


@tool
def updateOfferStatus(offerId: int, status: str) -> str:
    """Update the workflow status of an offer by id."""
    try:
        normalized = OfferStatus(status.strip().lower())
        dbOffer = updateOfferStatusRecord(offerId=offerId, status=normalized)
        return f"Offer status updated: id={dbOffer.id}, status={dbOffer.status.value}"
    except Exception as exc:
        return f"Error: updateOfferStatus failed: {exc}"


@tool
def updateOfferCvOutput(offerId: int, cvOutput: str) -> str:
    """Save generated CV output in an offer row and update offer status."""
    try:
        dbOffer = updateOfferCvOutputRecord(offerId=offerId, cvOutput=cvOutput)
        return f"Offer CV output updated: id={dbOffer.id}, status={dbOffer.status.value}"
    except Exception as exc:
        return f"Error: updateOfferCvOutput failed: {exc}"


@tool
def updateOfferCoverLetterOutput(offerId: int, coverLetterOutput: str) -> str:
    """Save generated cover letter output in an offer row and update offer status."""
    try:
        dbOffer = updateOfferCoverLetterOutputRecord(
            offerId=offerId,
            coverLetterOutput=coverLetterOutput,
        )
        return f"Offer cover letter output updated: id={dbOffer.id}, status={dbOffer.status.value}"
    except Exception as exc:
        return f"Error: updateOfferCoverLetterOutput failed: {exc}"
