import cv2, numpy as np, glob, json, sys
D=sys.argv[1]
fs=sorted(glob.glob(f"{D}/f/*.png")); fps=len(fs)/7.45
def masks(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); H,S,V=[hsv[:,:,i].astype(int) for i in range(3)]
    blue=((H>=95)&(H<=130)&(S>70)).astype(np.uint8)
    tan=((H>=8)&(H<=24)&(S>50)&(S<190)&(V>140)).astype(np.uint8)
    hair=((H<=20)&(S>90)&(V<140)&(V>25)).astype(np.uint8)
    red=(((H<=6)|(H>=170))&(S>150)&(V>60)).astype(np.uint8)
    return blue,tan,hair,red
def comps(m,minarea):
    n,lab,st,cen=cv2.connectedComponentsWithStats(m)
    return [(lab==i, st[i], cen[i]) for i in range(1,n) if st[i,4]>=minarea]
rows=[]; ref=None
for k,p in enumerate(fs):
    im=cv2.imread(p); blue,tan,hair,red=masks(im)
    # body = largest tan component with holes filled (face, heart, eyes)
    tc=max(comps(tan,1000), key=lambda c:c[1][4])
    body=np.zeros_like(tan); cnts,_=cv2.findContours(tc[0].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(body,cnts,-1,1,-1)
    ys,xs=np.nonzero(body); cx,cy=xs.mean(),ys.mean()
    top,bot=ys.min(),ys.max()
    # tilt: ellipse fitted to the convex hull of all large tan pieces (arms in front can split the body)
    pts=np.vstack([np.column_stack(np.nonzero(c[0])[::-1]) for c in comps(tan,1000)])
    hull=cv2.convexHull(pts.astype(np.int32))
    (ex,ey),(ea,eb),eang=cv2.fitEllipse(hull)
    tilt=eang if eang<90 else eang-180   # opencv: angle of the first axis; the long axis is near vertical
    if ea>eb: tilt = eang-90 if eang<180 else eang-270
    body=np.zeros_like(tan); cv2.drawContours(body,[hull],-1,1,-1)
    ys,xs=np.nonzero(body); cx,cy=xs.mean(),ys.mean(); top,bot=ys.min(),ys.max()
    if ref is None: ref={"H":bot-top,"W":xs.max()-xs.min(),"cx":cx,"cy":cy}
    Hh=ref["H"]
    # eyes: small blue blobs inside the body region
    eye=[c for c in comps(blue,80) if c[1][4]<3000 and body[int(c[2][1]),int(c[2][0])]]
    face_x=np.mean([c[2][0] for c in eye]) if len(eye)==2 else np.nan
    arms=[c for c in comps(blue,3000)]
    out={"t":k/fps,"cx":(cx-ref["cx"])/Hh,"cy":-(cy-ref["cy"])/Hh,"tilt":float(tilt),
         "height":(bot-top)/Hh,"width":(xs.max()-xs.min())/ref["W"],
         "turn":(face_x-cx)/(0.5*ref["W"]) if not np.isnan(face_x) else None}
    hy,hx=np.nonzero(hair)
    out["hair_dx"]=float((hx.mean()-cx)/Hh) if len(hx) else None
    # arms: shoulder anchors on the body side ~45% down; tip = farthest arm pixel from its anchor
    ang={}
    vis=cv2.addWeighted(im,0.6,np.zeros_like(im),0.4,0)
    for side,sgn in (("R",-1),("L",+1)):   # character's right arm is on image left
        ax=cx+sgn*0.40*ref["W"]; ay=top+0.47*Hh
        cand=[c for c in arms if (c[2][0]-cx)*sgn>0] or []
        if not cand: ang[side]=None; continue
        m=max(cand,key=lambda c:c[1][4])[0]
        yy,xx=np.nonzero(m); d=(xx-ax)**2+(yy-ay)**2; i=d.argmax()
        tx,ty=xx[i],yy[i]
        # raise: 0 = hanging down, 90 = out sideways, 180 = straight up; + across body beyond vertical is >180
        raise_=np.degrees(np.arctan2(sgn*(tx-ax), ty-ay))  # angle from straight down toward its own side
        ang[side]=float(raise_)
        cv2.line(vis,(int(ax),int(ay)),(int(tx),int(ty)),(0,255,255),4); cv2.circle(vis,(int(tx),int(ty)),10,(0,0,255),-1)
        cv2.putText(vis,f"{side} {raise_:.0f}",(int(tx)-30,int(ty)-15),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    out["arm_R"]=ang["R"]; out["arm_L"]=ang["L"]
    L=0.5*Hh; th=np.radians(tilt)
    cv2.line(vis,(int(cx),int(cy)),(int(cx+L*np.sin(th)),int(cy-L*np.cos(th))),(0,255,0),4)
    cv2.putText(vis,f"t={k/fps:.2f} tilt={tilt:.0f} turn={out['turn'] if out['turn'] is None else round(out['turn'],2)}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imwrite(f"{D}/ann/{k:04d}.png",cv2.resize(vis,(320,320)))
    rows.append(out)
json.dump({"fps":fps,"rows":rows},open(f"{D}/measure.json","w"))
print("done",len(rows))
