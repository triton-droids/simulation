"""Policy-first training workspace with embedded local results."""
import json
import time
from pathlib import Path
from robocub.user_cli.dashboard import Dashboard, ROOT, read_json
from robocub.user_cli.policy_library import policies, checkpoints, inside
from robocub.user_cli.training_recipe import build_recipe, save_recipe


class Workspace(Dashboard):
    def __init__(self, service=None):
        self.saved_recipe = None
        self.imported_config = None
        self.capture = None
        self.playing = False
        super().__init__(service)
        self.build_results()
        self.plans.master.pack_forget()
        self.root.geometry('1280x850')
        self.root.protocol('WM_DELETE_WINDOW',self.close)

    def build_policy_tab(self):
        tk,ttk = self.tk,self.ttk
        frame = self.policy_tab
        frame.columnconfigure(0,weight=1); frame.columnconfigure(1,weight=1)
        frame.rowconfigure(0,weight=1)
        left=ttk.Frame(frame); left.grid(row=0,column=0,sticky='nsew',padx=(0,10))
        self.policy_tree=ttk.Treeview(left,columns=('step','saved'),show='tree headings',selectmode='browse',height=9)
        self.policy_tree.heading('#0',text='Policy / checkpoint'); self.policy_tree.column('#0',width=235,minwidth=140)
        for key,title,width in [('step','Saved step',105),('saved','Saved UTC',160)]:
            self.policy_tree.heading(key,text=title); self.policy_tree.column(key,width=width,minwidth=65)
        scroll=ttk.Scrollbar(left,command=self.policy_tree.yview); scroll.pack(side='right',fill='y')
        self.policy_tree.configure(yscrollcommand=scroll.set); self.policy_tree.pack(fill='both',expand=True)
        self.tree_rows={}; self.policy_entries=policies(self.service.root)
        for index,entry in enumerate(self.policy_entries):
            key='policy'+str(index)
            rows=checkpoints(self.service.root,entry)
            self.policy_tree.insert('','end',iid=key,text=entry.get('display_name',entry['name']),open=index==0,
                values=(f"{rows[0]['step']:,}" if rows else 'No files', rows[0]['saved'] if rows else ''))
            self.tree_rows[key]=(entry, rows[0] if rows else None)
            for n,row in enumerate(rows):
                child=key+':'+str(row['step']); self.tree_rows[child]=(entry,row)
                self.policy_tree.insert(key,'end',iid=child,text=('Latest checkpoint' if n==0 else 'Checkpoint')+(' (missing files)' if not row['ready'] else ''),values=(f"{row['step']:,}",row['saved']))
        self.policy_tree.bind('<<TreeviewSelect>>',self.choose_tree)
        right=ttk.Frame(frame); right.grid(row=0,column=1,sticky='nsew')
        self.policy_name=tk.StringVar(); self.policy_status=tk.StringVar(); self.loaded_label=tk.StringVar()
        ttk.Label(right,textvariable=self.policy_status,wraplength=390).pack(anchor='w')
        ttk.Label(right,textvariable=self.loaded_label,wraplength=390).pack(anchor='w',pady=4)
        actions=ttk.Frame(right); actions.pack(fill='x')
        ttk.Button(actions,text='Watch checkpoint',command=self.watch).pack(side='left')
        ttk.Button(actions,text='Results / recordings',command=self.open_results).pack(side='left',padx=4)
        tabs=ttk.Notebook(right); tabs.pack(fill='both',expand=True,pady=5)
        basic=ttk.Frame(tabs,padding=5); advanced=ttk.Frame(tabs,padding=5)
        tabs.add(basic,text='Training settings'); tabs.add(advanced,text='Advanced')
        self.mode=tk.StringVar(value='Continue weights')
        ttk.Combobox(basic,textvariable=self.mode,values=['New training','Continue weights'],state='readonly').grid(row=0,column=0,columnspan=2,sticky='ew')
        self.fields={}
        self.record_video=tk.BooleanVar(value=False)
        labels=[('num_envs','Parallel environments'),('seed','Random seed'),('num_timesteps','Additional training steps'),('learning_rate','Learning rate'),('hours','Maximum hours')]
        more=[('batch_size','Batch size'),('num_minibatches','Minibatches'),('unroll_length','Rollout length'),('num_evals','Evaluation/checkpoint count'),('discounting','Discount'),('entropy_cost','Entropy cost'),('cpus','CPUs'),('memory_gb','RAM (GiB)')]
        for pane,items,start in [(basic,labels,1),(advanced,more,0)]:
            for n,(key,label) in enumerate(items,start):
                ttk.Label(pane,text=label).grid(row=n,column=0,sticky='w',pady=2)
                var=tk.StringVar(); self.fields[key]=var
                ttk.Entry(pane,textvariable=var,width=16).grid(row=n,column=1,sticky='ew',padx=5)
        ttk.Button(basic,text='Load checkpoint settings',command=self.load_settings).grid(row=6,column=0,columnspan=2,sticky='ew',pady=3)
        ttk.Button(basic,text='Import recipe JSON',command=self.import_recipe).grid(row=7,column=0,columnspan=2,sticky='ew')
        ttk.Checkbutton(basic,text='Record training evaluation video',variable=self.record_video).grid(row=8,column=0,columnspan=2,sticky='w')
        footer=ttk.Frame(frame); footer.grid(row=1,column=0,columnspan=2,sticky='ew',pady=5)
        ttk.Button(footer,text='Save training recipe',command=self.save_training).pack(side='left')
        ttk.Button(footer,text='Register reviewed recipe',command=self.register_recipe).pack(side='left',padx=4)
        self.train_button=ttk.Button(footer,text='Launch saved recipe',command=self.launch_saved); self.train_button.pack(side='left',padx=5)
        ttk.Button(footer,text='Show checkpoint files',command=self.open_checkpoint).pack(side='left')
        self.recipe_status=tk.StringVar(value='Training paused. Recipes can be edited and saved; launches remain blocked.')
        ttk.Label(frame,textvariable=self.recipe_status,wraplength=900).grid(row=2,column=0,columnspan=2,sticky='w')
        if (self.service.root/'Artifacts/Results/local_rb07_recovery/USER_PAUSED.txt').exists():
            self.train_button.configure(state='disabled')
        if self.tree_rows:
            key=next(iter(self.tree_rows)); self.policy_tree.selection_set(key); self.choose_tree()

    def choose_tree(self,event=None):
        selected=self.policy_tree.selection()
        if not selected:return
        self.policy_entry,self.current_checkpoint=self.tree_rows[selected[0]]
        self.policy_name.set(self.policy_entry.get('display_name',self.policy_entry['name']))
        self.policy_status.set(self.policy_name.get()+'\n'+self.policy_entry.get('status',''))
        self.loaded_checkpoint=self.current_checkpoint if self.current_checkpoint and self.current_checkpoint['ready'] else None
        row=self.current_checkpoint
        self.loaded_label.set(f"Total trained: {row['total']:,}\n{row['saved']} ({row['timestamp_source']})" if row else 'No local checkpoints')
        self.saved_recipe=None
        self.load_settings()

    def selected_checkpoint(self):return self.current_checkpoint

    def load_settings(self):
        self.imported_config=None
        self.saved_recipe=None
        cfg=read_json(inside(self.service.root,self.policy_entry['configuration']))
        agent=cfg.get('agent',{})
        defaults=dict(hours=1,cpus=3,memory_gb=16,num_timesteps=1003520)
        for key,var in self.fields.items():var.set(str(agent.get(key,defaults.get(key,''))))
        # Step budget is a new request, never silently repeat the historical full budget.
        self.fields['num_timesteps'].set('1003520')

    def import_recipe(self):
        from tkinter import filedialog
        name=filedialog.askopenfilename(filetypes=[('Training recipe','*.json')])
        if not name:return
        try:
            data=json.loads(Path(name).read_text(encoding='utf-8-sig'))
            self.imported_config=data.get('config',data) if 'agent' in data.get('config',data) else None
            agent=data.get('config',data).get('agent',data)
            for key,var in self.fields.items():
                value=agent.get(key,data.get('resources',{}).get(key,data.get(key)))
                if value is not None:var.set(str(value))
            self.saved_recipe=None
            self.recipe_status.set('Settings imported. Save a new recipe to validate them; no job launched.')
        except Exception as exc:self.messagebox.showerror('Import failed',str(exc))

    def save_training(self):
        try:
            recipe=build_recipe(self.service.root,self.policy_entry,self.current_checkpoint,
                {k:v.get() for k,v in self.fields.items()},self.source_id(),self.mode.get(),base_config=self.imported_config)
            recipe['record_video']=self.record_video.get()
            self.saved_recipe=save_recipe(self.service.root,recipe)
            self.recipe_status.set('Saved '+self.saved_recipe['plan']+' | Not launched. Commit/review this recipe before launch. Training remains paused.')
        except Exception as exc:self.messagebox.showerror('Invalid recipe',str(exc))

    def register_recipe(self):
        if not self.saved_recipe:
            self.messagebox.showinfo('Save first','Save a training recipe before registering it.');return
        entry=dict(self.saved_recipe,enabled=True)
        if not self.messagebox.askyesno('Register reviewed recipe?',f"Register {entry['id']} for {entry['source']}? This does not launch or lift the training pause. Commit the reviewed recipe and catalog before submission."):
            return
        path=self.service.root/'Configs/dashboard_plans.json'
        data=read_json(path)
        data.setdefault('plans',[])
        if any(e['id']==entry['id'] for e in data['plans']):
            self.messagebox.showinfo('Already registered','This recipe is already registered.');return
        data['plans'].append(entry)
        path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        self.recipe_status.set('Registered '+entry['id']+'. Commit reviewed changes before launch. Training pause is unchanged.')

    def launch_saved(self):
        if (self.service.root/'Artifacts/Results/local_rb07_recovery/USER_PAUSED.txt').exists():
            self.messagebox.showinfo('Training paused','Training remains paused by your request. This panel cannot clear that pause.');return
        if not self.saved_recipe:
            self.messagebox.showinfo('Save recipe','Save and review your settings first.');return
        # Existing launch service binds the exact approved plan bytes and checks clean source.
        entry=next((e for e in self.service.approved(self.source_id()) if e['id']==self.saved_recipe['id']),None)
        if not entry:
            self.messagebox.showinfo('Recipe not registered','Register this reviewed recipe in Configs/dashboard_plans.json and commit it before launching. The saved recipe folder includes the registration entry.');return
        if self.messagebox.askyesno('Launch training?',f"Launch {entry['id']} on {entry['source']}? Saved plan settings are fixed; current unsaved form edits are not included."):
            self.worker(lambda:self.service.launch(entry))

    def source_changed(self):
        super().source_changed()
        if hasattr(self,'saved_recipe'): self.saved_recipe=None

    def build_results(self):
        ttk,tk=self.ttk,self.tk
        self.results_tab=ttk.Frame(self.tabs,padding=8);self.tabs.add(self.results_tab,text='Results & recordings')
        self.results_tab.columnconfigure(1,weight=1);self.results_tab.rowconfigure(1,weight=1)
        self.result_location=tk.StringVar()
        ttk.Label(self.results_tab,textvariable=self.result_location,wraplength=900).grid(row=0,column=0,columnspan=2,sticky='w')
        self.result_list=tk.Listbox(self.results_tab,width=36,exportselection=False)
        self.result_list.grid(row=1,column=0,sticky='nsew')
        self.result_list.bind('<<ListboxSelect>>',self.show_result)
        self.result_view=ttk.Frame(self.results_tab);self.result_view.grid(row=1,column=1,sticky='nsew',padx=8)
        self.preview=ttk.Label(self.result_view,anchor='center');self.preview.pack(fill='both',expand=True)
        self.result_text=tk.Text(self.result_view,wrap='word',height=12)
        controls=ttk.Frame(self.results_tab);controls.grid(row=2,column=0,columnspan=2,sticky='ew')
        ttk.Button(controls,text='Play / pause video',command=self.toggle_video).pack(side='left')
        self.seek=tk.DoubleVar(value=0)
        self.seekbar=ttk.Scale(controls,from_=0,to=1,variable=self.seek,command=self.seek_video);self.seekbar.pack(side='left',fill='x',expand=True,padx=8)
        ttk.Button(controls,text='Refresh results',command=self.open_results).pack(side='right')

    def open_results(self):
        self.stop_video()
        run=self.selected() if self.tabs.select()==str(self.table.master) else None
        folder=run.folder if run else inside(self.service.root,self.policy_entry['run_dir'])
        self.result_location.set('Local saved results: '+str(folder)+' (remote-only files must be downloaded first)')
        # A training folder's siblings contain evaluation clips and screening results.
        base=folder.parent if folder.name=='train' else folder
        self.result_files=[]
        for path in base.rglob('*'):
            if any(p in ('checkpoints','source','src','.git') for p in path.relative_to(base).parts):continue
            if path.is_file() and path.suffix.lower() in ('.mp4','.png','.jpg','.jpeg','.json','.jsonl','.log','.csv'):
                self.result_files.append(path)
        self.result_files.sort(key=lambda p:(p.suffix not in ('.mp4','.png','.jpg'),str(p)))
        self.result_list.delete(0,'end')
        for path in self.result_files:self.result_list.insert('end',path.relative_to(base).as_posix())
        self.tabs.select(self.results_tab)
        if self.result_files:self.result_list.selection_set(0);self.show_result()
        else:self.show_text('No local recordings or results found for this policy. No simulation or download was started.')

    def recordings(self):
        self.stop_video()
        records=read_json(self.service.root/'Artifacts/Videos/index.json')
        self.result_files=[]
        for record in records if isinstance(records,list) else []:
            try:
                path=inside(self.service.root,record['path'])
                if path.is_file():self.result_files.append(path)
            except (KeyError,ValueError):continue
        self.result_location.set('All indexed recordings — paths identify the experiment; not necessarily the selected policy.')
        self.result_list.delete(0,'end')
        for path in self.result_files:self.result_list.insert('end',path.relative_to(self.service.root).as_posix())
        self.tabs.select(self.results_tab)
        if self.result_files:self.result_list.selection_set(0);self.show_result()
        else:self.show_text('No indexed recordings. Build the library with Scripts/index_videos.py.')


    def show_text(self,text):
        self.preview.pack_forget();self.result_text.pack(fill='both',expand=True)
        self.result_text.configure(state='normal');self.result_text.delete('1.0','end');self.result_text.insert('end',text);self.result_text.configure(state='disabled')

    def show_result(self,event=None):
        selected=self.result_list.curselection()
        if not selected:return
        self.stop_video();path=self.result_files[selected[0]]
        try:
            if path.suffix.lower() in ('.mp4','.webm','.avi','.mov','.mkv'):
                import cv2
                self.capture=cv2.VideoCapture(str(path))
                if not self.capture.isOpened():raise ValueError('Cannot decode this recording')
                self.fps=self.capture.get(cv2.CAP_PROP_FPS) or 25
                self.seekbar.configure(to=max(1,int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))-1))
                self.draw_video()
            elif path.suffix.lower() in ('.png','.jpg','.jpeg'):
                from PIL import Image
                self.display_image(Image.open(path))
            else:
                with path.open('rb') as stream:
                    stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-150000));text=stream.read().decode('utf-8',errors='replace')
                self.show_text(('Showing last 150 KB\n' if size>150000 else '')+text)
        except Exception as exc:self.show_text('Preview unavailable: '+str(exc))

    def display_image(self,image):
        from PIL import ImageTk
        self.result_text.pack_forget();self.preview.pack(fill='both',expand=True)
        image.thumbnail((max(320,self.result_view.winfo_width()),max(200,self.result_view.winfo_height())))
        self.preview_image=ImageTk.PhotoImage(image);self.preview.configure(image=self.preview_image)

    def draw_video(self):
        if self.capture is None:return
        import cv2
        from PIL import Image
        ok,frame=self.capture.read()
        if not ok:self.playing=False;return
        self.display_image(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
        self.seek.set(self.capture.get(cv2.CAP_PROP_POS_FRAMES)-1)
        if self.playing:self.video_timer=self.root.after(max(1,int(1000/self.fps)),self.draw_video)

    def toggle_video(self):
        if self.capture is None:return
        self.playing=not self.playing
        if self.playing:self.draw_video()
        elif getattr(self,'video_timer',None):self.root.after_cancel(self.video_timer);self.video_timer=None

    def seek_video(self,value):
        if self.capture is not None:
            import cv2
            self.capture.set(cv2.CAP_PROP_POS_FRAMES,float(value))
            if not self.playing:self.draw_video()

    def stop_video(self):
        self.playing=False
        if getattr(self,'video_timer',None):self.root.after_cancel(self.video_timer);self.video_timer=None
        if self.capture is not None:self.capture.release();self.capture=None

    def close(self):
        self.stop_video();self.root.destroy()
