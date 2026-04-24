"""
Example: Using the Auto-Evolution System

This example demonstrates:
- Recording task outcomes
- Detecting patterns
- Auto-generating skills
- Periodic reminders
"""

from open_mythos.evolution import AutoEvolution, TaskOutcome


def main():
    print("=== Auto-Evolution System Example ===\n")
    
    # Create evolution system
    evolution = AutoEvolution()
    
    # Record task outcomes
    print("1. Recording task outcomes...")
    
    tasks = [
        ("Fixed auth bug", TaskOutcome.SUCCESS, "Used TDD approach", 0.9),
        ("Added new feature", TaskOutcome.SUCCESS, "Used TDD approach", 0.85),
        ("Refactored code", TaskOutcome.SUCCESS, "Used TDD approach", 0.88),
        ("Database migration", TaskOutcome.SUCCESS, "Used TDD approach", 0.92),
        ("API endpoint", TaskOutcome.SUCCESS, "Used TDD approach", 0.87),
    ]
    
    for task, outcome, approach, quality in tasks:
        evolution.record_outcome(task, outcome, approach, quality)
        print(f"   - Recorded: {task}")
    
    # Check statistics
    print("\n2. System statistics:")
    
    stats = evolution.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # Check for new skills
    print("\n3. Checking for new skills...")
    
    new_skills = evolution.check_for_new_skill()
    
    print(f"   Found {len(new_skills)} potential new skills:")
    for skill in new_skills:
        print(f"   - {skill.name}")
        print(f"     Pattern: {skill.pattern[:50]}...")
        print(f"     Confidence: {skill.confidence:.2%}")
        print(f"     Success rate: {skill.success_rate:.2%}")
    
    # Get recent tasks
    print("\n4. Recent tasks:")
    
    recent = evolution.get_recent_tasks(limit=3)
    for task in recent:
        print(f"   - [{task.outcome.value}] {task.task}")
        print(f"     Approach: {task.approach}, Quality: {task.quality_score:.2f}")
    
    # Pattern detection
    print("\n5. Approach patterns detected:")
    
    patterns = evolution.pattern_detector.detect_approach_patterns()
    for name, count, success_rate in patterns[:5]:
        print(f"   - '{name}': {count} tasks, {success_rate:.1%} success")
    
    # Periodic reminders
    print("\n6. Periodic reminders:")
    
    # Add a reminder
    reminder = evolution.reminder_system.add_reminder(
        message="Review and update TDD skill",
        interval_hours=24,
        reminder_type="skill_review"
    )
    print(f"   Added reminder: {reminder.message}")
    print(f"   Next due: in {reminder.interval_hours} hours")
    
    # Check due reminders
    due = evolution.reminder_system.check_reminders()
    print(f"   Due reminders: {len(due)}")
    
    print("\n7. Evolution complete!")
    print("   The system is continuously learning from your tasks.")


if __name__ == "__main__":
    main()
