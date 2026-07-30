(function () {
  const script = document.currentScript;
  if (!script || document.querySelector(".qontic-feedback-widget")) return;

  const FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSfcN7XVUfAy6ge1Oc18mDCuNgx3fYwrTK_5huLncWxouob8Aw/viewform";
  const FIELD_ID = "entry.146585779";
  const FIELD_TITLE = "entry.1379107371";
  const FIELD_URL = "entry.1510255129";
  const siteRoot = new URL(".", script.src);

  function formUrl(id, title, pageUrl) {
    const url = new URL(FORM_URL);
    url.searchParams.set("usp", "pp_url");
    url.searchParams.set(FIELD_ID, id);
    url.searchParams.set(FIELD_TITLE, title);
    url.searchParams.set(FIELD_URL, pageUrl);
    return url.href;
  }

  const targets = [{
    label: script.dataset.feedbackKind === "module" ? "This module" : "This resource",
    id: script.dataset.feedbackId,
    title: script.dataset.feedbackTitle || document.title,
    url: window.location.href
  }];

  if (script.dataset.moduleId && script.dataset.feedbackKind !== "module") {
    targets.push({
      label: "Its module",
      id: script.dataset.moduleId,
      title: script.dataset.moduleTitle || script.dataset.moduleId,
      url: new URL(`modules/${encodeURIComponent(script.dataset.moduleId)}/`, siteRoot).href
    });
  }

  const style = document.createElement("style");
  style.textContent = `
    .qontic-feedback-widget{position:fixed;right:18px;bottom:18px;z-index:2147483000;font:14px/1.3 "Segoe UI",system-ui,sans-serif}
    .qontic-feedback-button{border:1px solid rgba(255,255,255,.35);border-radius:999px;background:#0055aa;color:#fff;padding:10px 15px;font:inherit;font-weight:700;box-shadow:0 4px 14px rgba(0,0,0,.25);cursor:pointer}
    .qontic-feedback-button:hover,.qontic-feedback-button:focus{background:#003f7f;outline:2px solid #8bc4ff;outline-offset:2px}
    .qontic-feedback-menu{position:absolute;right:0;bottom:48px;min-width:190px;padding:6px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;box-shadow:0 6px 20px rgba(0,0,0,.2)}
    .qontic-feedback-menu a{display:block;padding:9px 10px;border-radius:5px;color:#003366;text-decoration:none;font-weight:600;white-space:nowrap}
    .qontic-feedback-menu a:hover,.qontic-feedback-menu a:focus{background:#eef5fc;outline:none}
    @media(max-width:600px){.qontic-feedback-widget{right:10px;bottom:10px}.qontic-feedback-button{padding:9px 13px}}
  `;
  document.head.appendChild(style);

  const widget = document.createElement("div");
  widget.className = "qontic-feedback-widget";
  const button = document.createElement("button");
  button.type = "button";
  button.className = "qontic-feedback-button";
  button.textContent = "Feedback";
  widget.appendChild(button);

  function openTarget(target) {
    if (typeof window.gtag === "function") {
      window.gtag("event", "feedback_open", { feedback_id: target.id, feedback_title: target.title });
    }
    window.open(formUrl(target.id, target.title, target.url), "_blank", "noopener");
  }

  if (targets.length === 1) {
    button.addEventListener("click", () => openTarget(targets[0]));
  } else {
    const menu = document.createElement("div");
    menu.className = "qontic-feedback-menu";
    menu.hidden = true;
    targets.forEach(target => {
      const link = document.createElement("a");
      link.href = formUrl(target.id, target.title, target.url);
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = target.label;
      link.addEventListener("click", () => {
        if (typeof window.gtag === "function") {
          window.gtag("event", "feedback_open", { feedback_id: target.id, feedback_title: target.title });
        }
        menu.hidden = true;
      });
      menu.appendChild(link);
    });
    widget.insertBefore(menu, button);
    button.setAttribute("aria-haspopup", "true");
    button.setAttribute("aria-expanded", "false");
    button.addEventListener("click", () => {
      menu.hidden = !menu.hidden;
      button.setAttribute("aria-expanded", String(!menu.hidden));
    });
    document.addEventListener("click", event => {
      if (!widget.contains(event.target)) {
        menu.hidden = true;
        button.setAttribute("aria-expanded", "false");
      }
    });
  }

  document.body.appendChild(widget);
})();
