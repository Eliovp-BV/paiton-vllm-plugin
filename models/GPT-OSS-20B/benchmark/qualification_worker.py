"""Read-only worker telemetry for local qualification; never exposed publicly."""
class QualificationWorker:
    def qualification_memory(self):
        import torch
        torch.cuda.synchronize()
        return {
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        }
