import re

with open('train/run_biot_pretraining_resume.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the broken section: else block followed immediately by @torch.no_grad() without backward logic
# We need to insert backward logic between the end of train_one_epoch and @torch.no_grad()
pattern = r'''(        else:
            emb_pred, emb_clean = model\(x\)
            loss = F\.mse_loss\(emb_pred, emb_clean\))


@torch\.no_grad\(\)
def evaluate\(model, dataloader, device, use_amp=False, projector=None\):'''

replacement = r'''\1

        # Backward
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0 or step == num_steps - 1:
            print(f"  Epoch [{epoch}][{step}/{num_steps}] "
                  f"Loss: {loss.item():.4f}  LR: {optimizer.param_groups[0]['lr']:.6f}")

    return {'loss': total_loss / num_steps}


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp=False, projector=None):'''

content = re.sub(pattern, replacement, content)

with open('train/run_biot_pretraining_resume.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed!")
