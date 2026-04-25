"""
gpu_utils.py — GPU 管理 / 显存监控 / 错误保护
"""
import gc, time, traceback, torch

_R="\033[91m";_G="\033[92m";_Y="\033[93m";_C="\033[96m";_BD="\033[1m";_RS="\033[0m"

def list_gpus(show_processes=False):
    if not torch.cuda.is_available():
        print(f"{_Y}[Warning]{_RS} 未检测到 GPU。"); return []
    n=torch.cuda.device_count(); gpus=[]
    print(f"\n{_BD}┌{'─'*70}┐{_RS}")
    print(f"{_BD}│ {'ID':<4}{'型号':<28}{'空闲':>7}{'已用':>7}{'总计':>7}{'占用':>6} 状态{_RS}")
    print(f"{_BD}├{'─'*70}┤{_RS}")
    for i in range(n):
        pr=torch.cuda.get_device_properties(i)
        fb,tb=torch.cuda.mem_get_info(i); ub=tb-fb
        fg,ug,tg=fb/1024**3,ub/1024**3,tb/1024**3; pct=ug/tg*100
        c,st=(_G,"空闲 ✓") if fg>12 else ((_Y,"部分占用") if fg>4 else (_R,"繁忙 ✗"))
        nm=pr.name[:28]
        print(f"{_BD}│{_RS} [{i}]  {nm:<28} {c}{fg:>5.1f}GB{_RS}  {ug:>5.1f}GB  {tg:>5.1f}GB  {c}{pct:>4.0f}%{_RS}  {st}")
        gpus.append(dict(id=i,name=pr.name,free_gb=fg,used_gb=ug,total_gb=tg,pct=pct))
    print(f"{_BD}└{'─'*70}┘{_RS}")
    if show_processes:
        try:
            import pynvml,psutil; pynvml.nvmlInit(); any_p=False
            for i in range(n):
                h=pynvml.nvmlDeviceGetHandleByIndex(i)
                ps=pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                if ps:
                    if not any_p: print(f"\n{_BD}占用进程：{_RS}"); any_p=True
                    for p in ps:
                        try: proc=psutil.Process(p.pid); nm2=proc.name()[:20]; usr=proc.username()[:12]
                        except: nm2,usr="unknown","unknown"
                        mem=(p.usedGpuMemory or 0)/1024**3
                        print(f"  GPU {i}  PID {p.pid:<8}  {nm2:<22} {usr:<14} {mem:.1f}GB")
            if not any_p: print(f"\n  {_G}当前无其他进程占用 GPU。{_RS}")
        except ImportError:
            print(f"\n  {_Y}安装 pynvml+psutil 可查看占用进程{_RS}")
    return gpus

def select_gpu(gpu_id="auto", min_free_gb=4.0):
    if not torch.cuda.is_available():
        print(f"{_Y}无 GPU，返回 cpu{_RS}"); return "cpu"
    n=torch.cuda.device_count()
    if gpu_id=="auto":
        bi,bf=0,0.0
        for i in range(n):
            fb,_=torch.cuda.mem_get_info(i); fg=fb/1024**3
            if fg>bf: bf,bi=fg,i
        gpu_id=bi; print(f"{_G}[auto]{_RS} 选择 GPU {gpu_id}（空闲 {bf:.1f} GB）")
    gpu_id=int(gpu_id)
    if gpu_id>=n: raise ValueError(f"GPU {gpu_id} 不存在（共 {n} 个，0~{n-1}）")
    fb,tb=torch.cuda.mem_get_info(gpu_id); fg,tg=fb/1024**3,tb/1024**3
    pr=torch.cuda.get_device_properties(gpu_id)
    c=_G if fg>12 else (_Y if fg>4 else _R)
    print(f"\n{_BD}已选 GPU {gpu_id}：{pr.name}{_RS}")
    print(f"  空闲：{c}{fg:.1f} GB{_RS} / {tg:.0f} GB （已用 {tg-fg:.1f} GB）")
    if fg<min_free_gb: print(f"  {_R}⚠  空闲 {fg:.1f}GB < 建议最低 {min_free_gb}GB，建议换 GPU。{_RS}")
    torch.cuda.set_device(gpu_id); return f"cuda:{gpu_id}"

def get_mem_info(device):
    if not torch.cuda.is_available() or str(device)=="cpu":
        return dict(free=0,used=0,total=0,reserved=0,allocated=0,pct_used=0)
    idx=int(str(device).split(":")[-1]) if ":" in str(device) else 0
    fb,tb=torch.cuda.mem_get_info(idx); ub=tb-fb
    return dict(free=fb/1024**3,used=ub/1024**3,total=tb/1024**3,
                reserved=torch.cuda.memory_reserved(idx)/1024**3,
                allocated=torch.cuda.memory_allocated(idx)/1024**3,
                pct_used=ub/tb*100)

def vram_str(device):
    m=get_mem_info(device)
    if m["total"]==0: return "CPU"
    c=_G if m["pct_used"]<60 else (_Y if m["pct_used"]<85 else _R)
    return f"{c}{m['reserved']:.1f}/{m['total']:.0f}G{_RS}"

def print_oom_report(device, model_name="", params=None):
    m=get_mem_info(device)
    idx=int(str(device).split(":")[-1]) if ":" in str(device) else 0
    pr=torch.cuda.get_device_properties(idx) if torch.cuda.is_available() else None
    print(f"\n{'━'*60}")
    print(f"{_R}{_BD}  💥  CUDA Out of Memory  ─  {model_name}{_RS}")
    print(f"{'━'*60}")
    if pr: print(f"\n  {_BD}GPU {idx}：{pr.name}{_RS}")
    print(f"  {'显存总量':<18}  {m['total']:>6.2f} GB")
    print(f"  {'系统已占用':<16}  {m['used']:>6.2f} GB")
    print(f"  {'PyTorch 分配':<16}  {m['allocated']:>6.2f} GB")
    print(f"  {'PyTorch 缓存':<16}  {m['reserved']:>6.2f} GB")
    print(f"  {_R}{'真实空闲（OOM时）':<14}  {m['free']:>6.2f} GB{_RS}")
    n=torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n>1:
        print(f"\n  {_BD}其他 GPU 可用情况：{_RS}")
        for i in range(n):
            if str(i)==str(device).split(":")[-1]: continue
            fb,tb=torch.cuda.mem_get_info(i); fg=fb/1024**3
            pn=torch.cuda.get_device_properties(i).name[:28]
            c=_G if fg>12 else (_Y if fg>4 else _R)
            print(f"    GPU {i}  {pn}  {c}{fg:.1f} GB 空闲{_RS}")
    heads=(params or {}).get("gat_heads",4)
    print(f"\n  {_BD}三级优化建议：{_RS}")
    print(f"  {_G}① AMP（推荐）{_RS}  PARAMS[\"use_amp\"]=True  → 显存减少 ~45%")
    print(f"  {_Y}② 减少 GAT 头{_RS}  PARAMS[\"gat_heads\"]={max(1,heads//2)}  → 再减 ~30%")
    print(f"  {_Y}③ 梯度检查点{_RS}  PARAMS[\"use_ckpt\"]=True  → 速度慢 20%，但显存极省")
    print(f"  {_R}④ 切换 GPU{_RS}   修改 Cell 1 里 GPU_ID，换上面空闲更多的 GPU")
    print(f"{'━'*60}\n")

def clear_cache(device=None, verbose=True):
    bf=get_mem_info(device)["free"] if device else 0
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    if device and verbose:
        af=get_mem_info(device)
        print(f"  {_C}[cache]{_RS} 释放 {af['free']-bf:.2f} GB  空闲：{af['free']:.1f}/{af['total']:.0f} GB")

def safe_train(train_fn, *args, model_name="model", device="cuda:0",
               params=None, fallback=None, **kwargs):
    t0=time.time(); m0=get_mem_info(device)
    sep="─"*55
    print(f"\n{_BD}{sep}{_RS}")
    print(f"{_BD}  开始训练：{model_name}{_RS}")
    print(f"  显存：已用 {m0['used']:.1f} GB / 共 {m0['total']:.0f} GB  空闲 {m0['free']:.1f} GB")
    print(f"{_BD}{sep}{_RS}")
    try:
        result=train_fn(*args,**kwargs)
        el=time.time()-t0; h,m,s=int(el//3600),int(el%3600//60),int(el%60)
        m1=get_mem_info(device)
        print(f"\n  {_G}✅ {model_name} 完成{_RS}  用时 {h:02d}:{m:02d}:{s:02d}  显存：{m1['used']:.1f}/{m1['total']:.0f} GB")
        return result
    except torch.cuda.OutOfMemoryError:
        clear_cache(device,verbose=False)
        print_oom_report(device,model_name,params)
        print(f"  {_Y}→ {model_name} 跳过，后续 Cell 继续运行。{_RS}"); return fallback
    except RuntimeError as e:
        msg=str(e)
        if "out of memory" in msg.lower():
            clear_cache(device,verbose=False); print_oom_report(device,model_name,params)
        else:
            print(f"\n{_R}{'━'*55}{_RS}\n{_R}{_BD}  ❌ RuntimeError：{model_name}{_RS}\n  {msg}")
            traceback.print_exc(); print(f"{_R}{'━'*55}{_RS}"); clear_cache(device,verbose=True)
        print(f"  {_Y}→ {model_name} 跳过，后续 Cell 继续运行。{_RS}"); return fallback
    except KeyboardInterrupt:
        clear_cache(device,verbose=False)
        el=time.time()-t0; h,m,s=int(el//3600),int(el%3600//60),int(el%60)
        print(f"\n  {_Y}⚡ {model_name} 中断  用时 {h:02d}:{m:02d}:{s:02d}{_RS}")
        print(f"  {_Y}→ 返回 fallback，后续 Cell 继续运行。{_RS}"); return fallback
    except Exception as e:
        print(f"\n{_R}{'━'*55}{_RS}\n{_R}{_BD}  ❌ {type(e).__name__}：{model_name}{_RS}\n  {e}")
        traceback.print_exc(); print(f"{_R}{'━'*55}{_RS}")
        clear_cache(device,verbose=True)
        print(f"  {_Y}→ {model_name} 跳过，后续 Cell 继续运行。{_RS}"); return fallback
