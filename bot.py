"""? RLBot v5 controller. policy/ holds the whole policy as ONNX programs (convert_policy.py makes them from
policy.pt2): per-car state reconstruction, observation, the policies, the matchups they play and the action table.
This file copies packet fields into the tables the manifest names, runs its programs on CPU with onnxruntime and
starts each decided command at its chosen tick.
"""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from packet import BUTTONS, PacketReader

HERE=Path(__file__).resolve().parent
FORMAT='question_mark_policy_v2'

def load_policy(path):
    """policy/ -> (manifest, {program name: ONNX file})."""
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('format')!=FORMAT:raise ValueError(f"policy is {manifest.get('format')}, this bot runs {FORMAT}")
    return manifest,{name:path/f'{name}.onnx' for name in {manifest['step'],*manifest['matchups'].values()}}

class Programs:
    """The policy's programs as onnxruntime sessions over one table buffer: the state step every packet, each
    matchup's program per decision for this controller's cars."""
    def __init__(self,manifest,programs,sizes,threads=2):
        a,b,c,d=np.cumsum(sizes)
        self.buffer=np.zeros(d+1,np.float32)
        self.tables={'players':self.buffer[:a].reshape(int(manifest['max_players']),-1),'ball':self.buffer[a:b],
                     'pads':self.buffer[b:c].reshape(len(manifest['pad_locations']),-1),'game':self.buffer[c:d],
                     'dt':self.buffer[d:].reshape(())}
        self.state=np.zeros((int(manifest['max_players']),int(manifest['state_width'])),np.float32)
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.sessions={name:ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
                       for name,path in programs.items()}
        self.step_name=manifest['step']
    def reset(self):
        self.state[:]=0.
    def ready(self):
        """The buffer for this packet's tables and dt."""
        return self.buffer
    def step(self):
        """ready()'s buffer holds this packet's tables and dt: advance the state."""
        t=self.tables
        self.state=self.sessions[self.step_name].run(None,{'players':t['players'],'state':self.state,'dt':t['dt']})[0]
    def decide(self,name,rows):
        """After step(): run program `name` for packet rows; returns its decision rows."""
        t=self.tables
        feed={'players':t['players'],'state':self.state,'ball':t['ball'],'pads':t['pads'],'game':t['game'],
              'rows':np.asarray(rows,np.int64)}
        return self.sessions[name].run(None,feed)[0].tolist()

class BotController:
    def __init__(self,field_info,team,indices):
        manifest,programs=load_policy(HERE/'policy')
        self.manifest=manifest;self.label=manifest.get('label','?')
        self.interval=int(manifest['decision_interval']);self.active=set(manifest['active_phases'])
        self.matchups=manifest['matchups'];columns=manifest['decision_columns']
        self.delay_col=columns.index('start_delay_ticks');self.action_col=columns.index('action')
        self.control_cols=[columns.index(name) for name in manifest['controls']]
        self.reader=PacketReader(manifest,field_info)
        self.team=team;self.indices=list(indices)
        self.held=np.zeros((int(manifest['max_players']),len(manifest['controls'])),np.float32)
        self.pending={};self.next_decision=None;self.last_frame=None;self.last_phase=None;self.read_frame=None;self.decisions=[]
        self.programs=Programs(manifest,programs,self.reader.sizes)
    def step(self,packet):
        frame=int(packet.match_info.frame_num);phase=int(packet.match_info.match_phase)
        if self.last_frame == frame and self.last_phase == phase:
            return {i:self.held[i].copy() for i in self.indices}
        reset=self.last_frame is not None and (frame<self.last_frame or frame-self.last_frame>120)
        active=phase in self.active
        if not active or reset:
            self.programs.reset();self.read_frame=None;self.held.fill(0);self.pending.clear();self.next_decision=None
        self.last_frame=frame;self.last_phase=phase
        dt=0. if self.read_frame is None else (frame-self.read_frame)/120.;self.read_frame=frame
        deciding=active and (self.next_decision is None or frame>=self.next_decision)
        # A new decision replaces commands that have not started yet (even overdue ones, when a packet was skipped):
        # the car keeps the controls it already has until the new command's start tick.
        if not deciding:
            for i,(when,controls) in list(self.pending.items()):
                if frame>=when:self.held[i]=controls;del self.pending[i]
        if active:
            buffer=self.programs.ready();self.reader.read(packet,self.held,buffer[:-1]);buffer[-1]=dt
            self.programs.step()
            if deciding:
                own=sum(int(p.team)==self.team for p in packet.players);matchup=f'{own}v{len(packet.players)-own}'
                if matchup not in self.matchups:raise ValueError(f'the policy has no program for {matchup}')
                decision=self.programs.decide(self.matchups[matchup],self.indices)
                values=[int(d[self.action_col]) for d in decision]
                # Ticks count game time at any frame rate: each command starts its chosen delay after the decision's
                # frame, and the next decision comes a full interval after it, as in training.
                for i,d in zip(self.indices,decision):
                    self.pending[i]=(frame+int(d[self.delay_col]),np.array([d[k] for k in self.control_cols],np.float32))
                self.next_decision=frame+self.interval;self.decisions.append({'frame':frame,'encoded':values})
                if len(self.decisions)>2048:del self.decisions[:1024]
            for i,(when,controls) in list(self.pending.items()):
                if frame>=when:self.held[i]=controls;del self.pending[i]
        return {i:self.held[i].copy() for i in self.indices}

def make_hivemind_class():
    from rlbot.managers import Hivemind
    from rlbot import flat
    class Bot(Hivemind):
        def initialize(self):
            self.controller=BotController(self.field_info,self.team,self.indices)
            self._logger.info(f'{self.controller.label}: ONNX policy on CPU')
        def get_outputs(self,packet):
            controls=self.controller.step(packet);names=self.controller.manifest['controls']
            return {i:flat.ControllerState(**{n:bool(v) if n in BUTTONS else float(v) for n,v in zip(names,c)}) for i,c in controls.items()}
    return Bot

def main():
    make_hivemind_class()('skye/question_mark').run(wants_match_communications=False,wants_ball_predictions=False)
if __name__=='__main__':main()
