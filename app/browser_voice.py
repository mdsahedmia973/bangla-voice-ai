"""Browser SpeechSynthesis preview — client-side only, NO server audio.

The app's real voice is the server-side TinyTTS model. This preview uses
the USER'S OWN DEVICE (phone/computer) speech engine through the browser's
SpeechSynthesis API: if the device has a Bengali voice it speaks the
script with it, otherwise it says so plainly. The preview CANNOT be
downloaded as audio and is NOT the app's real voice — the UI must label
it that way (see the skill: browser speech preview is a quick check only).
"""
import json


def browser_preview_html(text: str, rate: float = 1.0) -> str:
    """Return a self-contained HTML snippet (for st.components.v1.html).

    `text` is embedded as a JSON string so quotes/newlines cannot break
    the script. `rate` maps 1.0 = normal device speed.
    """
    safe_text = json.dumps(text or "", ensure_ascii=False)
    safe_rate = float(max(0.5, min(2.0, rate or 1.0)))
    return f"""<div style="font-family: sans-serif; color: #e2e8f0;">
<button id="bnPreviewBtn" style="background:#7c6cff;color:#fff;border:0;
border-radius:8px;padding:10px 16px;font-size:15px;cursor:pointer;
width:100%;">&#128266; ব্রাউজার ভয়েসে শুনুন (প্রিভিউ)</button>
<p id="bnPreviewStatus" style="font-size:13px;margin:8px 0 0;color:#cbd5e1;">
ডিভাইসের বাংলা ভয়েস থাকলে এখান থেকে শোনা যাবে।</p>
<script>
(function(){{
  var text = {safe_text};
  var rate = {safe_rate};
  var btn = document.getElementById('bnPreviewBtn');
  var status = document.getElementById('bnPreviewStatus');
  function pickVoice(){{
    var voices = window.speechSynthesis.getVoices();
    var bn = voices.filter(function(v){{
      return (v.lang || '').toLowerCase().indexOf('bn') === 0;
    }});
    return {{all: voices, bn: bn}};
  }}
  function speak(){{
    if (!('speechSynthesis' in window)) {{
      status.textContent = 'এই ব্রাউজারে ভয়েস প্রিভিউ সমর্থিত নয়।';
      return;
    }}
    if (!text || !text.trim()) {{
      status.textContent = 'প্রথমে স্ক্রিপ্ট লিখুন।';
      return;
    }}
    window.speechSynthesis.cancel();
    var found = pickVoice();
    var u = new SpeechSynthesisUtterance(text);
    u.rate = rate;
    if (found.bn.length > 0) {{
      u.voice = found.bn[0];
      u.lang = found.bn[0].lang;
      status.textContent = 'বলছে — কণ্ঠ: ' + found.bn[0].name +
        ' (' + found.bn[0].lang + ')। এটা আপনার ডিভাইসের কণ্ঠ, অ্যাপের আসল কণ্ঠ নয়।';
    }} else {{
      u.lang = 'bn-IN';
      status.textContent = 'এই ডিভাইসে বাংলা ভয়েস পাওয়া যায়নি — ডিভাইসের ' +
        'ডিফল্ট কণ্ঠে বলার চেষ্টা করা হচ্ছে। এটা শুধু প্রিভিউ, ডাউনলোড করা যাবে না।';
    }}
    u.onend = function(){{
      status.textContent = 'প্রিভিউ শেষ হয়েছে। ডাউনলোডের জন্য উপরের "ভয়েস তৈরি করুন" ব্যবহার করুন।';
    }};
    u.onerror = function(e){{
      status.textContent = 'প্রিভিউ চালানো যায়নি (' + (e.error || 'অজানা ত্রুটি') + ')।';
    }};
    window.speechSynthesis.speak(u);
  }}
  btn.addEventListener('click', speak);
  if ('speechSynthesis' in window) {{
    window.speechSynthesis.onvoiceschanged = function(){{}};
    window.speechSynthesis.getVoices();
  }} else {{
    status.textContent = 'এই ব্রাউজারে ভয়েস প্রিভিউ সমর্থিত নয়।';
  }}
}})();
</script>
</div>"""
