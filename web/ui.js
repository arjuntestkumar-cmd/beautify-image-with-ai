/* ====================================================================
   Beautify — presentation layer only.

   This file is deliberately separate from app.js, which owns uploading,
   validation, polling, the result and the download. Nothing here touches
   any element app.js reads or writes, with one exception: the workspace
   compare slider's #cmp-after and #handle are left entirely alone here —
   the demo slider below uses its own #demo-* elements.
==================================================================== */
(() => {
  'use strict';

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const coarse = window.matchMedia('(pointer: coarse)').matches;

  /* ---------------------------------------------- scroll reveal ---- */
  const revealables = document.querySelectorAll('[data-reveal]');

  if (!('IntersectionObserver' in window) || reduced) {
    // Progressive enhancement: content must never be stuck invisible.
    document.documentElement.classList.add('no-reveal');
  } else {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        io.unobserve(entry.target);
      });
    }, { threshold: 0.18, rootMargin: '0px 0px -8% 0px' });

    revealables.forEach((el) => io.observe(el));

    // The how-it-works connector draws itself when the list appears.
    const steps = document.getElementById('steps');
    if (steps) {
      const lineIO = new IntersectionObserver((entries) => {
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          e.target.classList.add('is-visible');
          lineIO.unobserve(e.target);
        });
      }, { threshold: 0.3 });
      lineIO.observe(steps);
    }
  }

  /* ---------------------------------------------------- header ----- */
  const header = document.getElementById('site-header');
  const nav = document.getElementById('site-nav');
  const navToggle = document.getElementById('nav-toggle');
  const navLinks = [...document.querySelectorAll('.nav-link')];

  let ticking = false;
  const onScroll = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      header.classList.toggle('scrolled', window.scrollY > 12);
      highlight();
      ticking = false;
    });
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* active section indicator */
  const sections = navLinks
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  function highlight() {
    const line = window.scrollY + window.innerHeight * 0.32;
    let current = -1;
    sections.forEach((sec, i) => { if (sec.offsetTop <= line) current = i; });
    navLinks.forEach((a, i) => a.classList.toggle('active', i === current));
  }

  /* ------------------------------------------------ mobile menu ---- */
  if (navToggle && nav) {
    const setMenu = (open) => {
      nav.classList.toggle('open', open);
      navToggle.setAttribute('aria-expanded', String(open));
      navToggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
      // Stop the page scrolling behind the open menu on touch devices.
      document.body.classList.toggle('nav-open', open);
    };

    // A rotation to landscape can leave the menu open over a layout that no
    // longer collapses; close it whenever we cross back to the wide layout.
    window.addEventListener('resize', () => {
      if (window.innerWidth > 900) setMenu(false);
    }, { passive: true });
    navToggle.addEventListener('click', () =>
      setMenu(navToggle.getAttribute('aria-expanded') !== 'true'));
    nav.addEventListener('click', (e) => { if (e.target.closest('a')) setMenu(false); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setMenu(false); });
  }

  /* ------------------------------------------------------- FAQ ----- */
  document.querySelectorAll('.acc-q').forEach((btn) => {
    btn.addEventListener('click', () => {
      const item = btn.closest('.acc-item');
      const panel = item.querySelector('.acc-a');
      const open = btn.getAttribute('aria-expanded') === 'true';

      // close siblings for a single-open accordion
      document.querySelectorAll('.acc-item.open').forEach((other) => {
        if (other === item) return;
        other.classList.remove('open');
        other.querySelector('.acc-q').setAttribute('aria-expanded', 'false');
        other.querySelector('.acc-a').style.maxHeight = '';
      });

      btn.setAttribute('aria-expanded', String(!open));
      item.classList.toggle('open', !open);
      panel.style.maxHeight = open ? '' : `${panel.scrollHeight}px`;
    });
  });

  /* ---------------------------------------- comparison sliders ---- */
  /* Two of these on the page now - the illustrative one and the real sample - so the
     behaviour is written once and attached to whatever is present. Each slider owns its own
     ---split, so they drag independently. */
  const makeCompare = (root, after, handle) => {
    if (!root || !after || !handle) return;
    let dragging = false;
    const setSplit = (pct) => {
      const v = Math.max(0, Math.min(100, pct));
      after.style.setProperty('--split', `${v}%`);
      handle.style.left = `${v}%`;
      handle.setAttribute('aria-valuenow', Math.round(v));
    };
    const fromEvent = (e) => {
      const r = root.getBoundingClientRect();
      setSplit(((e.clientX - r.left) / r.width) * 100);
    };

    root.addEventListener('pointerdown', (e) => {
      dragging = true; root.setPointerCapture(e.pointerId); fromEvent(e);
    });
    root.addEventListener('pointermove', (e) => { if (dragging) fromEvent(e); });
    ['pointerup', 'pointercancel'].forEach((ev) =>
      root.addEventListener(ev, (e) => {
        dragging = false;
        try { root.releasePointerCapture(e.pointerId); } catch { /* already released */ }
      }));

    handle.addEventListener('keydown', (e) => {
      const now = Number(handle.getAttribute('aria-valuenow')) || 50;
      const step = e.shiftKey ? 10 : 3;
      if (e.key === 'ArrowLeft') { e.preventDefault(); setSplit(now - step); }
      if (e.key === 'ArrowRight') { e.preventDefault(); setSplit(now + step); }
      if (e.key === 'Home') { e.preventDefault(); setSplit(0); }
      if (e.key === 'End') { e.preventDefault(); setSplit(100); }
    });

    setSplit(50);
  };

  makeCompare(document.getElementById('hero-compare'),
              document.getElementById('hero-compare-after'),
              document.getElementById('hero-compare-handle'));
  makeCompare(document.getElementById('demo'),
              document.getElementById('demo-after'),
              document.getElementById('demo-handle'));

  /* ------------------------------------------- pointer-lit cards --- */
  if (!coarse && !reduced) {
    const lit = document.querySelectorAll('[data-tilt], .drop');
    lit.forEach((el) => {
      el.addEventListener('pointermove', (e) => {
        const r = el.getBoundingClientRect();
        el.style.setProperty('--mx', `${e.clientX - r.left}px`);
        el.style.setProperty('--my', `${e.clientY - r.top}px`);
      });
    });
  }

  /* ------------------------------------------------------- year ---- */
  const year = document.getElementById('year');
  if (year) year.textContent = String(new Date().getFullYear());
})();
