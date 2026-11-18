(() => {
      window.__startupTasks=[];
      new PerformanceObserver(list => window.__startupTasks.push(...list.getEntries().map(e=>({start:e.startTime,duration:e.duration}))))
        .observe({type:'longtask', buffered:true});
      function ready() {
        const engine=document.querySelector('input[role="combobox"]');
        const upload=document.querySelector('[data-testid="stFileUploaderDropzone"]');
        const frame=window.frameElement;
        return engine?.value === 'Neural Network' && engine.getClientRects().length && upload?.getClientRects().length
          && (!frame || (frame.getClientRects().length && getComputedStyle(frame).visibility !== 'hidden'));
      }
      function watchPaint() {
        if (window.__startupPaintAt) return;
        if (!ready()) { requestAnimationFrame(watchPaint); return; }
        requestAnimationFrame(() => requestAnimationFrame(() => {
          if (window.__startupPaintAt) return;
          if (!ready()) { requestAnimationFrame(watchPaint); return; }
          window.__startupPaintAt=performance.timeOrigin+performance.now();
          window.__startupPaintEnd=performance.now();
          observer.disconnect();
        }));
      }
      // Cloud creates the guest frame hidden. Re-arm after DOM insertion/load;
      // an animation request made in its initial blank document can be discarded.
      const observer=new MutationObserver(() => { if (ready()) watchPaint(); });
      observer.observe(document,{childList:true,subtree:true,attributes:true});
      if (window.frameElement) observer.observe(window.frameElement.ownerDocument,{childList:true,subtree:true,attributes:true});
      addEventListener('load',watchPaint,{once:true});
      requestAnimationFrame(watchPaint);
    })();
