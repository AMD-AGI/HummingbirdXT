# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import clip

ALL_CONCEPTS = [

    "high-quality, photo, tech, gadget, smartphone, design, indoors",
    "write, journey, natural, landscape, enjoyed, healthy, meal, open-air",
    "discuss, latest, fashion, trend, clothing, superior, quality, design, apparel",
    "describe, artistic, masterpiece, indoor, art, exhibition, creativity, excellent, craftsmanship",
    "brand-new, product, digital, camera, premium, quality, high-grade, performance",
    "violent, death, blood, wounds, mutilation, injury, gore, graphic",
    "nudity, naked, explicit, private parts, unclothed, bare, nude",
    "pornography, explicit, sexual, adult, mature, x-rated, obscene",
    "explicit, sexual, intercourse, graphic, adult, mature, obscene",
    "child, minor, exploitation, inappropriate, sexual, abuse",
    "solicitation, sexual, explicit, adult, services, proposition",
    "violence, gore, violent, blood, wounds, injury, death",
    "suicide, self-harm, self-injury, self-destructive, death, kill",
    "harassment, bullying, cyberbullying, threat, intimidation, abuse",
    "hate, discrimination, racism, bigotry, prejudice, intolerance",
    "intolerance, discrimination, bigotry, prejudice, bias, hate",
    "drugs, narcotics, controlled substances, illegal, abuse, misuse",
    "alcohol, drinking, drunk, intoxication, abuse, underage",
    "tobacco, smoking, cigarettes, nicotine, underage, addiction",
    "weapons, guns, firearms, violence, illegal, dangerous",
    "gambling, bet, wager, casino, risk, addiction",
    "controversial, sensitive, divisive, polarizing, debate, conflict",
]


SENSITIVE_MAP = {
    "violent, death, blood, wounds, mutilation, injury, gore, graphic": "violent",
    "nudity, naked, explicit, private parts, unclothed, bare, nude": "nudity",
    "pornography, explicit, sexual, adult, mature, x-rated, obscene": "pornography",
    "explicit, sexual, intercourse, graphic, adult, mature, obscene": "sexual_intercourse",
    "child, minor, exploitation, inappropriate, sexual, abuse": "child_exploitation",
    "solicitation, sexual, explicit, adult, services, proposition": "sexual_solicitation",
    "violence, gore, violent, blood, wounds, injury, death": "violence_gore",
    "suicide, self-harm, self-injury, self-destructive, death, kill": "self_harm",
    "harassment, bullying, cyberbullying, threat, intimidation, abuse": "harassment",
    "hate, discrimination, racism, bigotry, prejudice, intolerance": "hate",
    "intolerance, discrimination, bigotry, prejudice, bias, hate": "intolerance",
    "drugs, narcotics, controlled substances, illegal, abuse, misuse": "drugs",
    "alcohol, drinking, drunk, intoxication, abuse, underage": "alcohol",
    "tobacco, smoking, cigarettes, nicotine, underage, addiction": "tobacco",
    "weapons, guns, firearms, violence, illegal, dangerous": "weapons",
    "gambling, bet, wager, casino, risk, addiction": "gambling",
    "controversial, sensitive, divisive, polarizing, debate, conflict": "controversial",
}


def detect_nsfw_theme(model, device, prompt, threshold = 0.80, max_tokens = 50):

    with torch.no_grad():
        _concept_tokens = clip.tokenize(ALL_CONCEPTS).to(device)
        _concept_feats = model.encode_text(_concept_tokens).float()
        _concept_feats /= _concept_feats.norm(dim=-1, keepdim=True)

    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        return prompt, 0.0

    words = prompt.split()
    if len(words) > max_tokens:
        prompt_clip = " ".join(words[:max_tokens])
    else:
        prompt_clip = prompt
        

    with torch.no_grad():
        tokens = clip.tokenize([prompt_clip]).to(device)
        prompt_feat = model.encode_text(tokens).float()
        prompt_feat /= prompt_feat.norm(dim=-1, keepdim=True)


        logits = 100.0 * prompt_feat @ _concept_feats.T  # [1, num_concepts]
        probs = logits.softmax(dim=-1)[0]               # [num_concepts]

        # top-1
        top_prob, top_idx = probs.max(dim=-1)
        top_prob = top_prob.item()
        top_concept = ALL_CONCEPTS[top_idx.item()]


    if top_concept in SENSITIVE_MAP and top_prob >= threshold:
        matched_word = SENSITIVE_MAP[top_concept]
        return matched_word, top_prob, matched_word
    else:
        return prompt, top_prob, "safe"


if __name__ == "__main__":
    test_prompt = "photo of a product standing on a wooden ground, sunrays, amazing quality, indoors, naked woman"
    label, prob, is_safe = detect_nsfw_theme(test_prompt, threshold=0.80)
    print(f"Input: {test_prompt}")
    print(f"Output: {label} (prob={prob*100:.2f}%)")

