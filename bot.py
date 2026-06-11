import os
import base64
import asyncio
import logging
import httpx
from pathlib import Path
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TG_TOKEN       = os.getenv("TG_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ALLOWED_USERS  = set(os.getenv("ALLOWED_USERS", "").split(","))

DATA_DIR  = Path("data")
ROOMS_DIR = DATA_DIR / "rooms"
FACES_DIR = DATA_DIR / "faces"
for d in [ROOMS_DIR, FACES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

class Gen(StatesGroup):
    waiting_ref   = State()
    waiting_extra = State()
    generating    = State()
    feedback      = State()

class Upload(StatesGroup):
    waiting_room = State()
    waiting_face = State()

def load_images_b64(folder: Path) -> list:
    images = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            data = f.read_bytes()
            b64  = base64.b64encode(data).decode()
            mime = "image/jpeg" if f.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            images.append({"b64": b64, "mime": mime, "name": f.name})
    return images

async def download_tg_photo(bot: Bot, photo: types.PhotoSize) -> bytes:
    file = await bot.get_file(photo.file_id)
    buf  = await bot.download_file(file.file_path)
    return buf.read()

async def generate_prompt_gemini(ref_bytes: bytes, extra: str, rooms: list, faces: list) -> str:
    system_instruction = (
        "You are an expert AI image prompt engineer specializing in realistic lifestyle/fashion photography. "
        "Your task: analyze the reference photo and write a detailed English prompt for an AI image generator. "
        "Rules:\n"
        "- Start with 'a girl'\n"
        "- NO description of facial features or ethnicity\n"
        "- DO describe: exact pose, hand positions, body posture, mouth/lips expression, gaze direction, figure shape\n"
        "- DO describe: clothing color, style, fabric texture in detail\n"
        "- DO describe: shooting style (POV, mirror selfie, etc), camera angle\n"
        "- DO describe: lighting color, intensity, atmosphere, mood\n"
        "- DO describe: phone camera imperfections — slight blur, grain, uneven focus, natural noise\n"
        "- The background MUST be from the provided room photos — pick the most suitable room and describe it as the location\n"
        "- End with technical tags: 4k, realistic texture, photorealistic, shot on smartphone\n"
        "- Do NOT use markdown. Output ONLY the prompt text, nothing else."
    )

    parts = []
    ref_b64 = base64.b64encode(ref_bytes).decode()
    parts.append({"text": "REFERENCE PHOTO (recreate this composition/pose/vibe):"})
    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}})

    if rooms:
        parts.append({"text": f"\nAVAILABLE ROOMS ({len(rooms)} options — pick best match and use as background):"})
        for r in rooms[:6]:
            parts.append({"inline_data": {"mime_type": r["mime"], "data": r["b64"]}})
            parts.append({"text": f"[Room: {r['name']}]"})

    if faces:
        parts.append({"text": f"\nMODEL FACE REFERENCES ({len(faces)} photos):"})
        for f in faces[:4]:
            parts.append({"inline_data": {"mime_type": f["mime"], "data": f["b64"]}})

    if extra.strip():
        parts.append({"text": f"\nADDITIONAL INSTRUCTIONS: {extra}"})

    parts.append({"text": "\nNow write the prompt:"})

    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 800}
    }

    url = f"https://aiplatform.googleapis.com/v1/projects/my-bot-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

async def generate_image_gemini(prompt: str) -> bytes:
    url = f"https://aiplatform.googleapis.com/v1/projects/my-bot-project/locations/us-central1/publishers/google/models/gemini-2.0-flash-preview-image-generation:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]}
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    for part in data["candidates"][0]["content"]["parts"]:
        if part.get("inlineData"):
            return base64.b64decode(part["inlineData"]["data"])
    raise ValueError("Картинка не пришла от Gemini")

bot = Bot(token=TG_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())

def main_keyboard():
    return ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="🎨 Сгенерировать фото")],
        [KeyboardButton(text="🏠 Загрузить комнату"), KeyboardButton(text="👤 Загрузить лицо")],
        [KeyboardButton(text="📋 Моя база")],
    ], resize_keyboard=True)

def feedback_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Подходит, сохраняю", callback_data="ok")],
        [InlineKeyboardButton(text="🔄 Перегенерировать (тот же промт)", callback_data="regen")],
        [InlineKeyboardButton(text="✏️ Исправить и перегенерировать", callback_data="edit")],
        [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")],
    ])

def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USERS or ALLOWED_USERS == {''}:
        return True
    return str(user_id) in ALLOWED_USERS

@dp.message(Command("start"))
async def cmd_start(msg: types.Message, state: FSMContext):
    if not is_allowed(msg.from_user.id):
        await msg.answer("⛔ Нет доступа.")
        return
    await state.clear()
    rooms = load_images_b64(ROOMS_DIR)
    faces = load_images_b64(FACES_DIR)
    await msg.answer(
        f"👋 Привет! Я твой бот для генерации фото.\n\n"
        f"📦 База: <b>{len(rooms)}</b> комнат, <b>{len(faces)}</b> фото лица\n\n"
        f"Нажми <b>Сгенерировать фото</b> и кидай референс!",
        parse_mode="HTML",
        reply_markup=main_keyboard()
    )

@dp.message(F.text == "📋 Моя база")
async def cmd_base(msg: types.Message):
    rooms = load_images_b64(ROOMS_DIR)
    faces = load_images_b64(FACES_DIR)
    room_names = "\n".join(f"  • {r['name']}" for r in rooms) or "  (пусто)"
    face_names = "\n".join(f"  • {f['name']}" for f in faces) or "  (пусто)"
    await msg.answer(
        f"🏠 <b>Комнаты ({len(rooms)}):</b>\n{room_names}\n\n"
        f"👤 <b>Лица модели ({len(faces)}):</b>\n{face_names}",
        parse_mode="HTML"
    )

@dp.message(F.text == "🏠 Загрузить комнату")
async def upload_room_start(msg: types.Message, state: FSMContext):
    await state.set_state(Upload.waiting_room)
    await msg.answer("📸 Отправь фото комнаты (можно несколько подряд). Когда закончишь — напиши /done")

@dp.message(Upload.waiting_room, F.photo)
async def upload_room_photo(msg: types.Message, state: FSMContext, bot: Bot):
    photo = msg.photo[-1]
    data = await download_tg_photo(bot, photo)
    fname = ROOMS_DIR / f"room_{photo.file_unique_id}.jpg"
    fname.write_bytes(data)
    rooms = load_images_b64(ROOMS_DIR)
    await msg.answer(f"✅ Комната сохранена! Всего комнат: {len(rooms)}")

@dp.message(F.text == "👤 Загрузить лицо")
async def upload_face_start(msg: types.Message, state: FSMContext):
    await state.set_state(Upload.waiting_face)
    await msg.answer("📸 Отправь фото лица модели (можно несколько). Когда закончишь — напиши /done")

@dp.message(Upload.waiting_face, F.photo)
async def upload_face_photo(msg: types.Message, state: FSMContext, bot: Bot):
    photo = msg.photo[-1]
    data = await download_tg_photo(bot, photo)
    fname = FACES_DIR / f"face_{photo.file_unique_id}.jpg"
    fname.write_bytes(data)
    faces = load_images_b64(FACES_DIR)
    await msg.answer(f"✅ Фото лица сохранено! Всего фото: {len(faces)}")

@dp.message(Command("done"))
async def cmd_done(msg: types.Message, state: FSMContext):
    await state.clear()
    await msg.answer("👍 Загрузка завершена!", reply_markup=main_keyboard())

@dp.message(F.text == "🎨 Сгенерировать фото")
async def gen_start(msg: types.Message, state: FSMContext):
    await state.set_state(Gen.waiting_ref)
    await msg.answer("📸 Кидай фото-референс (что хочешь увидеть):")

@dp.message(Gen.waiting_ref, F.photo)
async def gen_got_ref(msg: types.Message, state: FSMContext, bot: Bot):
    photo = msg.photo[-1]
    data = await download_tg_photo(bot, photo)
    await state.update_data(ref_bytes=data)
    await state.set_state(Gen.waiting_extra)
    await msg.answer(
        "✏️ Напиши что хочешь исправить или добавить\n(или отправь <b>-</b> если всё ок):",
        parse_mode="HTML"
    )

@dp.message(Gen.waiting_extra, F.text)
async def gen_got_extra(msg: types.Message, state: FSMContext, bot: Bot):
    extra = "" if msg.text.strip() == "-" else msg.text.strip()
    data  = await state.get_data()
    ref_bytes = data.get("ref_bytes", b"")

    await state.set_state(Gen.generating)
    status = await msg.answer("⏳ Пишу промт через Gemini...")

    rooms = load_images_b64(ROOMS_DIR)
    faces = load_images_b64(FACES_DIR)

    try:
        prompt = await generate_prompt_gemini(ref_bytes, extra, rooms, faces)
    except Exception as e:
        logger.exception("Gemini prompt error")
        await status.edit_text(f"❌ Ошибка промта: {e}")
        await state.clear()
        return

    await status.edit_text(f"✅ Промт готов! Генерирую картинку...\n\n<code>{prompt[:600]}</code>", parse_mode="HTML")
    await state.update_data(last_prompt=prompt, extra=extra)

    try:
        img_bytes = await generate_image_gemini(prompt)
    except Exception as e:
        logger.exception("Gemini image error")
        await status.edit_text(f"❌ Ошибка генерации: {e}\n\nПромт:\n<code>{prompt[:600]}</code>", parse_mode="HTML")
        await state.clear()
        return

    await status.delete()
    await bot.send_photo(
        msg.chat.id,
        types.BufferedInputFile(img_bytes, filename="result.jpg"),
        caption=f"🎨 <b>Готово!</b>\n\n<code>{prompt[:600]}</code>",
        parse_mode="HTML",
        reply_markup=feedback_keyboard()
    )
    await state.set_state(Gen.feedback)

@dp.callback_query(Gen.feedback, F.data == "ok")
async def feedback_ok(call: types.CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup()
    await call.message.answer("🔥 Огонь! Готов к следующему.", reply_markup=main_keyboard())
    await state.clear()

@dp.callback_query(Gen.feedback, F.data == "regen")
async def feedback_regen(call: types.CallbackQuery, state: FSMContext, bot: Bot):
    await call.message.edit_reply_markup()
    data   = await state.get_data()
    prompt = data.get("last_prompt", "")
    status = await call.message.answer("🔄 Перегенерирую...")
    try:
        img_bytes = await generate_image_gemini(prompt)
    except Exception as e:
        await status.edit_text(f"❌ Ошибка: {e}")
        await state.clear()
        return
    await status.delete()
    await bot.send_photo(
        call.message.chat.id,
        types.BufferedInputFile(img_bytes, filename="result.jpg"),
        caption=f"🎨 <b>Перегенерировано</b>\n\n<code>{prompt[:600]}</code>",
        parse_mode="HTML",
        reply_markup=feedback_keyboard()
    )

@dp.callback_query(Gen.feedback, F.data == "edit")
async def feedback_edit(call: types.CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup()
    await call.message.answer("✏️ Напиши что исправить:")
    await state.set_state(Gen.waiting_extra)

@dp.callback_query(Gen.feedback, F.data == "cancel")
async def feedback_cancel(call: types.CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup()
    await call.message.answer("❌ Отменено.", reply_markup=main_keyboard())
    await state.clear()

@dp.message(F.photo)
async def any_photo(msg: types.Message, state: FSMContext):
    cur = await state.get_state()
    if cur in (Upload.waiting_room.state, Upload.waiting_face.state):
        return
    await msg.answer("Нажми <b>Сгенерировать фото</b> и потом кидай референс 👆", parse_mode="HTML", reply_markup=main_keyboard())

async def main():
    logger.info("Bot starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
